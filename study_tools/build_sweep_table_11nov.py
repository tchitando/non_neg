#!/usr/bin/env python3
"""Standalone sweep table (crop-only base, OCT, ResNet-50, validation), every seed 42-44, mean, SD, and the level the
selection rule picks for the heatmap: highest 1/2 (NCL mean + SimCLR mean) of validation probe h over seeds 42-44,
among levels with seeds 42-44 for both methods."""
import sys
import numpy as np
import pandas as pd

MASTER, OUT = sys.argv[1], sys.argv[2]
SEEDS = [42, 43, 44]
# operation, parameter, in heatmap, [(level, policy id, note)] from mild to strong
SWEEPS = [
    ("Brightness", "strength (p 0.8)", True, [
        ("0.2", "croponly_bright_b020", ""), ("0.4", "croponly_bright_b040", ""),
        ("0.8", "croponly_bright", ""), ("0.95", "croponly_bright_b095", "")]),
    ("Contrast", "factor (p 0.5); lower = stronger", True, [
        ("0.8", "croponly_contrast", ""), ("0.5", "croponly_contrast_f050", "seeds 45-46 also exist; left out"),
        ("0.3", "croponly_contrast_f030", ""), ("0.1", "croponly_contrast_f010", "")]),
    ("Gaussian blur", "sigma max, sigma ~ U[0.1, max] (p 0.5)", True, [
        ("0.6", "croponly_blur_sig060", ""), ("1.2", "croponly_blur", ""),
        ("2.0", "croponly_blur_sig200", "1-seed screen; seeds 43-44 deferred until after the heatmap")]),
    ("Flip", "horizontal flip (p 0.5); on/off only", True, [
        ("on", "croponly_flip", "no strength to choose")]),
    ("Speckle", "std (p 0.5)", True, [
        ("0.05", "croponly_speckle_s005", ""), ("0.1", "croponly_speckle", ""),
        ("0.2", "croponly_speckle_s020", "1-seed screen")]),
    ("JPEG", "quality (p 0.5); lower = stronger", True, [
        ("90", "croponly_jpeg_q90", ""),
        ("70", "croponly_jpeg", "severity screen borderline: worst-class drop 8.12 (run 1), 5.10 (run 2), tolerance 6"),
        ("40", "croponly_jpeg_q40", "1-seed screen")]),
    ("Pixelation", "factor (p 0.5); lower = stronger", False, [
        ("0.9", "croponly_pixelate_f090", "1-seed screen; seeds 43-44 deferred until after the heatmap"),
        ("0.8", "croponly_pixelate", "NCL s43 diverged (NaN at step 160 of epoch 0, reproduced on retrain)"),
        ("0.6", "croponly_pixelate_f060", "1-seed screen")]),
    ("Motion blur", "radius, sigma, +-15 deg (p 0.5)", False, [
        ("3, 1", "croponly_motion_r3", "1-seed screen"),
        ("5, 2", "croponly_motion", "SimCLR s42 did not finish; NCL s43 stopped at epoch 32; seeds deferred"),
        ("8, 3", "croponly_motion_r8", "s42 did not finish for either method")]),
    ("Crop min area", "RandomResizedCrop minimum area; base is 0.08", False, [
        ("0.2", "croponly_crop020", "crop is part of the base (0.08); not a heatmap operation"),
        ("0.5", "croponly_crop050", "NCL s42 diverged; 1-seed screen")]),
]

m = pd.read_csv(MASTER).drop_duplicates("run_id", keep="last")
m = m[m.run_id.str.contains("resnet50-octmnist")]
H = {(r.policy_id, r.objective, int(r.seed)): r.probe_bb_bal_acc for r in m.itertuples()}
SC = {(r.policy_id, r.objective, int(r.seed)): r.sc for r in m.itertuples()}


def stats(D, pid, obj):
    v = [D.get((pid, obj, s), np.nan) for s in SEEDS]
    ok = [x for x in v if np.isfinite(x)]
    mean = np.mean(ok) if ok else np.nan
    sd = np.std(ok, ddof=1) if len(ok) > 1 else np.nan
    return v, mean, sd, len(ok)


rows = []
def add(op, param, level, pid, note, score=np.nan, picked=""):
    hm = {}
    for obj, name in (("ncl", "NCL"), ("simclr", "SimCLR")):
        hv, hmean, hsd, n = stats(H, pid, obj)
        sv, smean, ssd, _ = stats(SC, pid, obj)
        hm[obj] = hmean
        rows.append({"Operation": op, "Parameter": param, "Level": level, "Policy ID": pid, "Method": name,
                     "n seeds": n,
                     **{f"Val h s{s}": hv[i] for i, s in enumerate(SEEDS)}, "Val h mean": hmean, "Val h SD": hsd,
                     **{f"Val SC % s{s}": sv[i] for i, s in enumerate(SEEDS)}, "Val SC % mean": smean, "Val SC % SD": ssd,
                     "Score 1/2 (NCL + SimCLR h mean)": score, "Picked for heatmap": picked, "Note": note})
    if np.isfinite(hm["ncl"]) and np.isfinite(hm["simclr"]):
        rows[-2]["NCL h mean above SimCLR"] = "yes" if hm["ncl"] > hm["simclr"] else "no"

add("Base (crop only)", "crop min area 0.08, nothing else", "-", "croponly",
    "reference, not a candidate; seeds 45-46 also exist; left out")
picks = {}
for op, param, inmap, levels in SWEEPS:
    sc = {}
    for level, pid, note in levels:
        full = all(np.isfinite(H.get((pid, o, s), np.nan)) for o in ("ncl", "simclr") for s in SEEDS)
        if full:
            sc[level] = 0.5 * (stats(H, pid, "ncl")[1] + stats(H, pid, "simclr")[1])
    best = max(sc, key=sc.get) if sc else None
    picks[op] = (best, sc)
    for level, pid, note in levels:
        if not inmap:
            pk = "not in heatmap"
        elif op == "Flip":
            pk = "yes (on)"
        else:
            pk = "yes" if level == best else ("" if level in sc else "not eligible (needs seeds 42-44, both methods)")
        if op == "Flip":
            s = sc.get(level, np.nan)
        else:
            s = sc.get(level, np.nan)
        add(op, param, level, pid, note, s, pk)

cols = ["Operation", "Parameter", "Level", "Policy ID", "Method", "n seeds",
        "Val h s42", "Val h s43", "Val h s44", "Val h mean", "Val h SD",
        "Val SC % s42", "Val SC % s43", "Val SC % s44", "Val SC % mean", "Val SC % SD",
        "NCL h mean above SimCLR", "Score 1/2 (NCL + SimCLR h mean)", "Picked for heatmap", "Note"]
t = pd.DataFrame(rows).reindex(columns=cols)
for c in cols:
    if c.startswith("Val") :
        t[c] = t[c].round(2)
t["Score 1/2 (NCL + SimCLR h mean)"] = t["Score 1/2 (NCL + SimCLR h mean)"].round(3)
t.to_csv(OUT, index=False)
for op, (b, sc) in picks.items():
    print(f"{op:15s} pick {b}  " + "  ".join(f"{k}: {v:.3f}" for k, v in sc.items()))
