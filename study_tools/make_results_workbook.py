#!/usr/bin/env python3
"""Results workbook: 1 Specialisation, 2 Performance, 3 Augmentations.

    python study_tools/make_results_workbook.py            # from the repository root
    -> study/summary/results_workbook.xlsx

Reads whatever of these exists (missing ones are listed on the Read me sheet):
    study/results_master.csv + results_spec.csv               OCT, ResNet-50 (validation)
    study/rn18/results_master.csv + results_spec.csv          OCT, ResNet-18 (validation)
    study/path/results_master.csv + results_spec.csv          PathMNIST, ResNet-50 (validation)
    study/test_results_octmnist.csv, study/test_results_pathmnist.csv   eval_test.py (test split)
    study/rn18/test_probe.log                                 ResNet-18 test probes (eval_linear_octmnist.py)

Sheets 1 and 2 show one row per policy with NCL and SimCLR side by side, means over seeds, for the main policies
(stem, single operations, all six, natural-image set, A6). "More policies" holds the rest of the screen. The two
Runs sheets hold one row per run. Every number is a plain value computed from the csv files (no spreadsheet formulas);
run the script again after new results arrive. Runs whose backbone
probe is below 60 (collapsed NCL runs on PathMNIST) are left out of the means.
"""
import math, re
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.comments import Comment

OUT = Path("study/summary/results_workbook.xlsx")
SOURCES = [("octmnist", "ResNet-50", Path("study")), ("octmnist", "ResNet-18", Path("study/rn18")),
           ("pathmnist", "ResNet-50", Path("study/path"))]
CLASSES = {"octmnist": 4, "pathmnist": 9}
DNAME = {"octmnist": "OCT", "pathmnist": "PathMNIST"}
POOL = {"octmnist": ["speckle", "contrast", "blur", "motion", "pixelate", "jpeg"],
        "pathmnist": ["contrast", "rotflip", "stain", "defocus", "pixelate", "jpeg"]}
OPNAME = {"speckle": "Speckle", "contrast": "Contrast", "blur": "Blur", "motion": "Motion", "pixelate": "Pixelation",
          "jpeg": "JPEG", "rotflip": "Rotation + flip", "stain": "Stain", "defocus": "Defocus"}
CROPONLY = {  # the crop-only base (Part 7) and its strength scan (Part 8): one augmentation at one strength each
    "croponly": "Crop only (base)", "croponly_bright": "Crop only + brightness 0.8 (stem's)",
    "croponly_bright_b020": "Crop only + brightness 0.2", "croponly_bright_b040": "Crop only + brightness 0.4",
    "croponly_bright_b095": "Crop only + brightness 0.95",
    "croponly_flip": "Crop only + flip (stem's)",
    "croponly_stem": "Crop only + brightness + flip (= stem)", "croponly_contrast": "Crop only + contrast 0.8",
    "croponly_contrast_f050": "Crop only + contrast 0.5", "croponly_contrast_f030": "Crop only + contrast 0.3",
    "croponly_contrast_f010": "Crop only + contrast 0.1", "croponly_blur_sig060": "Crop only + blur \u03c3 \u2264 0.6",
    "croponly_blur": "Crop only + blur \u03c3 \u2264 1.2", "croponly_blur_sig200": "Crop only + blur \u03c3 \u2264 2.0",
    "croponly_crop020": "Crop only, crop area 0.20", "croponly_crop050": "Crop only, crop area 0.50",
    "croponly_speckle_s005": "Crop only + speckle 0.05", "croponly_speckle": "Crop only + speckle 0.1",
    "croponly_speckle_s020": "Crop only + speckle 0.2", "croponly_motion_r3": "Crop only + motion 3, 1",
    "croponly_motion": "Crop only + motion 5, 2", "croponly_motion_r8": "Crop only + motion 8, 3",
    "croponly_pixelate_f090": "Crop only + pixelation 0.9", "croponly_pixelate": "Crop only + pixelation 0.8",
    "croponly_pixelate_f060": "Crop only + pixelation 0.6", "croponly_jpeg_q90": "Crop only + JPEG 90",
    "croponly_jpeg": "Crop only + JPEG 70", "croponly_jpeg_q40": "Crop only + JPEG 40",
    "croponly_proj2048": "Crop only, projector output 2048",
}
COLLAPSE_BELOW = 60.0
OBJ = [("ncl", "NCL"), ("simclr", "SimCLR")]

FONT = "Arial"
HEAD_FILL = PatternFill("solid", start_color="DCEBEC")
BLOCK_FILL = PatternFill("solid", start_color="F2F6F7")
STEM_FILL = PatternFill("solid", start_color="FBF6E9")
# analysis fills on the NCL rows of "0 Full table": NCL against SimCLR (beyond 2 SE = darker), the story mark, NCL SC level
CMP_FILL = {("up", True): "B9DFBF", ("up", False): "E6F4E7", ("down", False): "FBE5E2", ("down", True): "F2BDB6"}
STORY_FILL = "F5D77A"
SC_LO, SC_HI = (0xEE, 0xF4, 0xFB), (0x8F, 0xB8, 0xE0)
THIN = Side(style="thin", color="C9D3D6")
MUTED = "555F66"


def policy_info(pid, ds):
    """(display name, main sheet?, sort key) for a policy id."""
    pool = POOL[ds]
    if pid in ("stem", "A0"):
        return "Stem", True, (0, 0)
    if pid in CROPONLY:
        return CROPONLY[pid], True, (4.5, list(CROPONLY).index(pid))
    if pid in pool:
        return OPNAME[pid], True, (1, pool.index(pid))
    if pid == "all6":
        return "All six", True, (2, 0)
    if pid == "cifar":
        return "Natural-image set", True, (3, 0)
    if pid == "speckle_contrast_crop020":
        return "A6 (speckle + contrast, crop 0.2)", True, (4, 0)
    m = re.match(r"^(.*)_(crop0(\d\d)|f0(\d\d)|nocrop)$", pid)
    if m:
        base, tail = m.group(1), m.group(2)
        bname = "Stem" if base == "stem" else " + ".join(OPNAME.get(p, p).lower() for p in base.split("_")).capitalize()
        what = (f"crop area {int(tail[4:]) / 100:.2f}" if tail.startswith("crop") else "no crop" if tail == "nocrop"
                else f"contrast factor {int(tail[1:]) / 100:.2f}")
        return f"{bname}, {what}", False, (7, pid)
    parts = pid.split("_")
    if all(p in pool for p in parts):
        name = " + ".join(OPNAME[p].lower() for p in parts).capitalize()
        return name, False, ((5 if len(parts) == 2 else 6), tuple(pool.index(p) for p in parts))
    return pid, False, (8, pid)


def read_nonfinite():
    """Diverged runs listed by the scorers (study/nonfinite_runs.txt, study/path/nonfinite_runs.txt), if copied over."""
    out = set()
    for p in (Path("study/nonfinite_runs.txt"), Path("study/path/nonfinite_runs.txt")):
        if p.exists():
            out |= {ln.split("\t")[0] for ln in p.read_text().splitlines() if ln.strip()}
    return out


def read_policy_augs():
    """study/derived/policy_augs.csv (describe_policies.py): what each policy applied, read from its training files."""
    p = Path("study/derived/policy_augs.csv")
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    return {(r.dataset, r.backbone, r.policy_id): r.augmentations for r in d.itertuples()}


FULL_FONT = "Garamond"


PLAN_LABEL = {"test_plan_oct.csv": "27 Oct plan", "test_plan_path.csv": "27 Oct plan",
              "test_plan_croponly.csv": "2 Nov crop-only plan",
              "test_plan_rest_oct.csv": "added 2 Nov, after analysis began",
              "test_plan_rest_path.csv": "added 2 Nov, after analysis began"}


def test_plan_label(t):
    """Which test plan(s) a row's test runs were scored under (plans are fixed before scoring)."""
    if not len(t) or "plan" not in t.columns:
        return None
    labs = sorted({PLAN_LABEL.get(str(p).split("/")[-1], str(p).split("/")[-1]) for p in t["plan"].dropna()})
    return "; ".join(labs) if labs else None


def full_table(wb, val, tdf, keys, seeds):
    """Sheet '0 Full table' and study/summary/full_table.csv: one row per dataset, backbone, policy and method.
    Layout as edited by Tine (TC_results.xlsx, 2 Nov): header in row 1, block rows, per-seed probe, SC % and purity."""
    ws = wb.create_sheet("0 Full table", 1)
    augs = read_policy_augs()
    ok = val[~(val.probe_bb_bal_acc < COLLAPSE_BELOW)]

    def vals(d, col):
        return pd.to_numeric(d[col], errors="coerce").dropna() if len(d) and col in d.columns else pd.Series(dtype=float)

    def msd(d, col):
        x = vals(d, col)
        return (float(x.mean()) if len(x) else None, float(x.std(ddof=1)) if len(x) > 1 else None, len(x))

    def seed_val(d, col, sd):
        if not len(d) or col not in d.columns:
            return None
        x = pd.to_numeric(d[pd.to_numeric(d["seed"], errors="coerce") == sd][col], errors="coerce").dropna()
        return float(x.iloc[-1]) if len(x) else None

    def reference(pid):
        if pid in ("stem", "A0", "croponly"):
            return None
        return "croponly" if pid.startswith("croponly") else "stem"

    cols = [("Dataset", None), ("Backbone", None), ("Policy", None),
            ("Augmentations applied (level, probability per view)", None), ("Method", None), ("n seeds, validation", "0")]
    cols += [(f"Probe h, seed {sd}", "0.00") for sd in seeds]
    cols += [("Probe h mean", "0.00"), ("Probe h SD", "0.00"), ("Probe h minus reference", "+0.00;-0.00"),
             ("Probe z mean", "0.00"), ("Probe z SD", "0.00"), ("k-NN-20 mean", "0.00"), ("k-NN-20 SD", "0.00")]
    cols += [(f"SC %, seed {sd}", "0.0") for sd in seeds] + [("SC % mean", "0.0"), ("SC % SD", "0.0")]
    cols += [(f"Purity z, seed {sd}", "0.000") for sd in seeds] + [("Purity z mean", "0.000"), ("Purity z SD", "0.000")]
    cols += [("Purity z* mean", "0.000"), ("Purity h mean", "0.000"), ("Purity h* mean", "0.000")]
    cols += [(f"MS, seed {sd}", "0.0000") for sd in seeds] + [("MS mean", "0.0000"), ("MS SD", "0.0000"),
             ("MS floor (chance)", "0.0000"), ("MS* mean", "0.000"), ("Eff. dim z mean", "0.0"),
             ("n seeds, test", "0"), ("Test probe h mean", "0.00"), ("Test probe h SD", "0.00"), ("Test probe z mean", "0.00"),
             ("Test AUC h mean", "0.00"), ("Test k-NN-20 mean", "0.00"), ("Test SC* mean", "0.0"),
             ("Test purity z* mean", "0.000"), ("Reference", None), ("Test plan", None)]
    for j, (h, _f) in enumerate(cols, start=1):
        cellf(ws, 1, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=FULL_FONT)
        ws.cell(row=1, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[1].height = 42
    means = {}
    for k in keys.itertuples():
        for o, _on in OBJ:
            means[(k.dataset, k.backbone, k.policy_id, o)] = ok[(ok.dataset == k.dataset) & (ok.backbone == k.backbone)
                                                                & (ok.policy_id == k.policy_id) & (ok.objective == o)]
    rows_csv, r, prev, written = [], 2, None, []
    for k in keys.itertuples():
        block = (k.dataset, k.backbone, "crop-only" if k.policy_id.startswith("croponly") else "screen")
        for o, oname in OBJ:
            allr = val[(val.dataset == k.dataset) & (val.backbone == k.backbone) & (val.policy_id == k.policy_id) & (val.objective == o)]
            d = means[(k.dataset, k.backbone, k.policy_id, o)]
            t = (tdf[(tdf.dataset == k.dataset) & (tdf.backbone == k.backbone) & (tdf.policy_id == k.policy_id) & (tdf.objective == o)]
                 if len(tdf) else tdf)
            if allr.empty and len(t) == 0:
                continue
            if block != prev:
                title = (f"{DNAME[k.dataset]}, {k.backbone}" + (": crop-only design (reference: crop only)" if block[2] == "crop-only"
                         else (": old-stem screen (reference: stem)" if k.dataset == "octmnist" and k.backbone == "ResNet-50" else "")))
                for c in range(1, len(cols) + 1):
                    cellf(ws, r, c, fill=BLOCK_FILL, size=9, font=FULL_FONT)
                cellf(ws, r, 1, title, bold=True, fill=BLOCK_FILL, size=9, font=FULL_FONT)
                r += 1; prev = block
            ref = reference(k.policy_id)
            dref = means.get((k.dataset, k.backbone, ref, o)) if ref else None
            pb, pbs, n = msd(d, "probe_bb_bal_acc")
            refdiff = None
            if dref is not None and len(dref) and pb is not None:
                b = vals(dref, "probe_bb_bal_acc")
                refdiff = pb - float(b.mean()) if len(b) else None
            pz, pzs, _ = msd(d, "probe_proj_bal_acc"); kn, kns, _ = msd(d, "knn20_bal_acc")
            scm, scs, _ = msd(d, "sc_pct"); puz, puzs, _ = msd(d, "purity_proj")
            puzst = msd(d, "purity_proj_star")[0]; puhraw = msd(d, "purity_bb")[0]
            puh = msd(d, "purity_bb_star")[0]; ms = msd(d, "ms_star")[0]; ed = msd(d, "effdim_proj")[0]
            tb, tbs, tn = msd(t, "probe_bb_bal_acc"); tz = msd(t, "probe_proj_bal_acc")[0]; ta = msd(t, "probe_bb_auc")[0]
            tk = msd(t, "knn20_bal_acc")[0]; tsc = msd(t, "sc_star")[0]; tpu = msd(t, "purity_proj_star")[0]
            aug_pid = {"A0": "stem", "A6": "speckle_contrast_crop020", "croponly_stem": "stem"}.get(k.policy_id, k.policy_id)
            row = [DNAME[k.dataset], k.backbone, policy_info(k.policy_id, k.dataset)[0],
                   augs.get((DNAME[k.dataset], k.backbone, aug_pid)), oname, n]
            row += [seed_val(d, "probe_bb_bal_acc", sd) for sd in seeds]
            row += [pb, pbs, refdiff, pz, pzs, kn, kns]
            row += [seed_val(d, "sc_pct", sd) for sd in seeds] + [scm, scs]
            row += [seed_val(d, "purity_proj", sd) for sd in seeds] + [puz, puzs, puzst, puhraw]
            msr, msrs, _ = msd(d, "ms"); msf = msd(d, "ms_floor")[0]
            row += [puh] + [seed_val(d, "ms", sd) for sd in seeds] + [msr, msrs, msf]
            row += [ms, ed, tn or None, tb, tbs, tz, ta, tk, tsc, tpu,
                    {"croponly": "crop only", "stem": "stem"}.get(ref, "") if ref else "(reference)",
                    test_plan_label(t)]
            stem = k.policy_id in ("stem", "A0", "croponly")
            for j, (v, (_h, fmt)) in enumerate(zip(row, cols), start=1):
                cellf(ws, r, j, v, fmt=fmt, bold=(stem and j <= 5), fill=STEM_FILL if stem else None, size=9, font=FULL_FONT)
            rows_csv.append(dict(zip([h for h, _f in cols], row), policy_id=k.policy_id))
            written.append((r, k.dataset, k.backbone, k.policy_id, o))
            r += 1
    analysis_fills(ws, cols, written, ok, tdf, vals)
    widths = [9, 9, 34, 62, 7, 7] + [7] * (len(cols) - 6)
    for c, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(c)].width = w
    pd.DataFrame(rows_csv).to_csv(OUT.parent / "full_table.csv", index=False)


CMP = [("Probe h mean", "val", "probe_bb_bal_acc"), ("Probe z mean", "val", "probe_proj_bal_acc"),
       ("k-NN-20 mean", "val", "knn20_bal_acc"), ("Purity z mean", "val", "purity_proj"), ("Purity z* mean", "val", "purity_proj_star"),
       ("Purity h mean", "val", "purity_bb"), ("Purity h* mean", "val", "purity_bb_star"), ("MS mean", "val", "ms"), ("MS* mean", "val", "ms_star"),
       ("Test probe h mean", "test", "probe_bb_bal_acc"), ("Test probe z mean", "test", "probe_proj_bal_acc"),
       ("Test AUC h mean", "test", "probe_bb_auc"), ("Test k-NN-20 mean", "test", "knn20_bal_acc"),
       ("Test purity z* mean", "test", "purity_proj_star")]


FILL_KEY = [(None, "Fills on the NCL rows"),
    (CMP_FILL[("up", True)], "NCL above SimCLR, beyond 2 SE"),
    (CMP_FILL[("up", False)], "NCL above SimCLR, within noise"),
    (CMP_FILL[("down", False)], "NCL below SimCLR, within noise"),
    (CMP_FILL[("down", True)], "NCL below SimCLR, beyond 2 SE"),
    (STORY_FILL, "Policy: story holds on validation or test (NCL at least level with SimCLR on both probes, "
                 "higher purity); the note on the cell gives the split and the numbers"),
    ("BFD5ED", "SC: NCL's level within the dataset, darker = more specialised (SimCLR's SC is at chance by construction)"),
    (None, "SE from the seed SD pooled over the policies of that dataset, backbone, method and split; "
           "policies with one seed use the pooled SD.")]


def analysis_fills(ws, cols, written, ok, tdf, vals):
    """Fills on the NCL row of each policy. Each compared mean: green where NCL is above SimCLR, rose where below,
    darker where the difference is beyond 2 SE (seed SD pooled over the policies of that dataset, backbone, method and
    split; SE = sqrt(sd_ncl^2/n_ncl + sd_simclr^2/n_simclr)). Policy cell gold where the story holds on a split: NCL at
    least level with SimCLR on both the backbone and the projector probe, with higher projector purity (a note on the
    cell gives the numbers). SC cells shaded blue by NCL's level within the dataset (SimCLR's SC is at chance)."""
    hdr = {h: j for j, (h, _f) in enumerate(cols, start=1)}
    sdcache = {}

    def sel(src, ds, bb, pid, o):
        df = ok if src == "val" else tdf
        if not len(df):
            return df
        return df[(df.dataset == ds) & (df.backbone == bb) & (df.policy_id == pid) & (df.objective == o)]

    def pooled(src, ds, bb, o, col):
        key = (src, ds, bb, o, col)
        if key not in sdcache:
            df = ok if src == "val" else tdf
            sdcache[key] = None
            if len(df) and col in df.columns:
                s = df[(df.dataset == ds) & (df.backbone == bb) & (df.objective == o)]
                x = pd.DataFrame({"p": s["policy_id"].values, "v": pd.to_numeric(s[col], errors="coerce").values}).dropna()
                g = x.groupby("p")["v"]
                n, v = g.count(), g.var(ddof=1)
                m = n > 1
                dfree = float((n[m] - 1).sum())
                if dfree > 0:
                    sdcache[key] = math.sqrt(float(((n[m] - 1) * v[m]).sum()) / dfree)
        return sdcache[key]

    fmt_d = lambda col, d: (f"{d:+.3f}" if "purity" in col or "ms_" in col else f"{d:+.2f}")
    for r, ds, bb, pid, o in written:
        if o != "ncl":
            continue
        res = {}
        for h, src, col in CMP:
            a, b = vals(sel(src, ds, bb, pid, "ncl"), col), vals(sel(src, ds, bb, pid, "simclr"), col)
            if not len(a) or not len(b):
                continue
            diff = float(a.mean() - b.mean())
            sa, sb = pooled(src, ds, bb, "ncl", col), pooled(src, ds, bb, "simclr", col)
            se = math.sqrt(sa ** 2 / len(a) + sb ** 2 / len(b)) if sa is not None and sb is not None else None
            strong = se is not None and se > 0 and abs(diff) > 2 * se
            res[(src, col)] = (diff, strong, len(a), len(b))
            if diff != 0:
                ws.cell(row=r, column=hdr[h]).fill = PatternFill("solid", start_color=CMP_FILL[("up" if diff > 0 else "down", strong)])
        notes = []
        for src, name in (("val", "validation"), ("test", "test")):
            need = [(src, "probe_bb_bal_acc"), (src, "probe_proj_bal_acc"), (src, "purity_proj_star")]
            if all(k in res for k in need) and res[need[0]][0] >= 0 and res[need[1]][0] >= 0 and res[need[2]][0] > 0:
                (db, sb_, na, nb), (dz, sz, _, _), (dp, sp, _, _) = (res[k] for k in need)
                noise = lambda st: "beyond 2 SE" if st else "within noise"
                notes.append(f"Story holds on {name} ({na} NCL / {nb} SimCLR seeds): NCL minus SimCLR, backbone probe "
                             f"{db:+.2f} ({noise(sb_)}), projector probe {dz:+.2f} ({noise(sz)}), projector purity {dp:+.3f} ({noise(sp)}).")
        if notes:
            c = ws.cell(row=r, column=hdr["Policy"])
            c.fill = PatternFill("solid", start_color=STORY_FILL)
            c.comment = Comment("\n".join(notes), "make_results_workbook.py", width=320, height=110)
    # NCL's SC level, shaded within each dataset (validation SC %, test SC*)
    for h in ("SC % mean", "Test SC* mean"):
        j = hdr[h]
        by_ds = {}
        for r, ds, bb, pid, o in written:
            v = ws.cell(row=r, column=j).value
            if o == "ncl" and isinstance(v, (int, float)) and not math.isnan(v):
                by_ds.setdefault(ds, []).append((r, float(v)))
        for ds, rv in by_ds.items():
            lo, hi = min(v for _, v in rv), max(v for _, v in rv)
            for r, v in rv:
                f = 0.0 if hi <= lo else (v - lo) / (hi - lo)
                col = "".join(f"{round(a + f * (b - a)):02X}" for a, b in zip(SC_LO, SC_HI))
                ws.cell(row=r, column=j).fill = PatternFill("solid", start_color=col)
    # key, to the right of the table
    c0 = len(cols) + 2
    key = FILL_KEY
    for i, (fc, text) in enumerate(key, start=2):
        if fc:
            cellf(ws, i, c0, None, fill=PatternFill("solid", start_color=fc), size=9, font=FULL_FONT)
        cellf(ws, i, c0 + 1, text, bold=fc is None and i == 2, size=9, font=FULL_FONT)
    ws.column_dimensions[get_column_letter(c0)].width = 4
    ws.column_dimensions[get_column_letter(c0 + 1)].width = 60


def pooled_sd(df, ds, bb, o, col):
    """Seed SD pooled over the policies of one dataset, backbone and method (policies with 2+ seeds)."""
    if not len(df) or col not in df.columns:
        return None
    s = df[(df.dataset == ds) & (df.backbone == bb) & (df.objective == o)]
    x = pd.DataFrame({"p": s["policy_id"].values, "v": pd.to_numeric(s[col], errors="coerce").values}).dropna()
    g = x.groupby("p")["v"]
    n, v = g.count(), g.var(ddof=1)
    m = n > 1
    dfree = float((n[m] - 1).sum())
    return math.sqrt(float(((n[m] - 1) * v[m]).sum()) / dfree) if dfree > 0 else None


def ncl_ahead_sheet(wb, ok, tdf, keys):
    """Sheet '0b NCL ahead of SimCLR': every policy where NCL's mean is above SimCLR's on at least one linear probe
    (backbone h or projector z, validation or test), with both methods' means and SDs, the difference, its z
    (difference / SE, SE from the seed SD pooled over policies), NCL's SC and both methods' purity."""
    ws = wb.create_sheet("0b NCL ahead of SimCLR", 2)
    augs = read_policy_augs()
    F = FULL_FONT

    def sel(df, ds, bb, pid, o):
        if not len(df):
            return df
        return df[(df.dataset == ds) & (df.backbone == bb) & (df.policy_id == pid) & (df.objective == o)]

    def ms(d, col):
        x = pd.to_numeric(d[col], errors="coerce").dropna() if len(d) and col in d.columns else pd.Series(dtype=float)
        return (float(x.mean()) if len(x) else None, float(x.std(ddof=1)) if len(x) > 1 else None, len(x))

    probes = [("Probe h, validation", ok, "probe_bb_bal_acc", "val h"), ("Probe z, validation", ok, "probe_proj_bal_acc", "val z"),
              ("Probe h, test", tdf, "probe_bb_bal_acc", "test h"), ("Probe z, test", tdf, "probe_proj_bal_acc", "test z")]
    rows = []
    for k in keys.itertuples():
        ds, bb, pid = k.dataset, k.backbone, k.policy_id
        rec = {"k": k, "probe": {}, "ahead": []}
        for label, df, col, short in probes:
            a, b = ms(sel(df, ds, bb, pid, "ncl"), col), ms(sel(df, ds, bb, pid, "simclr"), col)
            if a[0] is None or b[0] is None:
                rec["probe"][label] = None
                continue
            sa, sb = pooled_sd(df, ds, bb, "ncl", col), pooled_sd(df, ds, bb, "simclr", col)
            se = math.sqrt(sa ** 2 / a[2] + sb ** 2 / b[2]) if sa is not None and sb is not None else None
            d = a[0] - b[0]
            rec["probe"][label] = (a, b, d, d / se if se else None)
            if d > 0:
                rec["ahead"].append(short)
        if not rec["ahead"]:
            continue
        n = ms(sel(ok, ds, bb, pid, "ncl"), "probe_bb_bal_acc")
        rec["sc"] = ms(sel(ok, ds, bb, pid, "ncl"), "sc_pct")
        rec["scab"] = ms(sel(ok, ds, bb, pid, "ncl"), "sc_above")
        rec["msv"] = ms(sel(ok, ds, bb, pid, "ncl"), "ms_star")
        rec["sct"] = ms(sel(tdf, ds, bb, pid, "ncl"), "sc_pct")
        rec["pu"] = (ms(sel(ok, ds, bb, pid, "ncl"), "purity_proj"), ms(sel(ok, ds, bb, pid, "simclr"), "purity_proj"))
        rec["put"] = (ms(sel(tdf, ds, bb, pid, "ncl"), "purity_proj_star"), ms(sel(tdf, ds, bb, pid, "simclr"), "purity_proj_star"))
        rec["nv"] = (n[2], ms(sel(ok, ds, bb, pid, "simclr"), "probe_bb_bal_acc")[2])
        rec["nt"] = (ms(sel(tdf, ds, bb, pid, "ncl"), "probe_bb_bal_acc")[2], ms(sel(tdf, ds, bb, pid, "simclr"), "probe_bb_bal_acc")[2])
        rows.append(rec)

    # header: title, rule, then two header rows (group, column)
    cellf(ws, 1, 1, "Policies where NCL's mean is above SimCLR's on at least one linear probe", bold=True, size=13, font=F)
    cellf(ws, 2, 1, "A row is listed when NCL's mean beats SimCLR's on the backbone (h) or projector (z) probe, on validation or test. "
                    "\"NCL ahead on\" says where. z = NCL minus SimCLR divided by its SE (seed SD pooled over policies); "
                    "|z| > 2 = beyond noise (darker fill). SD is over seeds (empty with one seed). SimCLR's SC sits at chance "
                    "by construction (its units are never zero) and its MS* is near zero (about 0.001), so both are shown for NCL only; "
                    "MS* is scored on validation only. Purity reads both methods. Collapsed and diverged runs are left out.",
          size=9, italic=True, font=F, color=MUTED)
    ident = ["Dataset", "Backbone", "Policy", "Augmentations applied (level, probability per view)",
             "Seeds, validation (NCL / SimCLR)", "Seeds, test (NCL / SimCLR)", "NCL ahead on"]
    pcols = ["NCL mean", "NCL SD", "SimCLR mean", "SimCLR SD", "NCL − SimCLR", "z"]
    groups = [(None, ident)] + [(lab, pcols) for lab, *_ in probes] + [
        ("NCL SC %, validation", ["Mean", "SD"]), ("NCL SC above shuffle, validation", ["Mean", "SD"]),
        ("NCL MS*, validation", ["Mean", "SD"]),
        ("NCL SC %, test", ["Mean", "SD"]),
        ("Purity z (raw), validation", ["NCL mean", "NCL SD", "SimCLR mean", "SimCLR SD", "NCL − SimCLR"]),
        ("Purity z*, test", ["NCL mean", "SimCLR mean", "NCL − SimCLR"])]
    hr1, hr2 = 4, 5
    c = 1
    for g, subs in groups:
        if g:
            ws.merge_cells(start_row=hr1, start_column=c, end_row=hr1, end_column=c + len(subs) - 1)
            cellf(ws, hr1, c, g, bold=True, fill=HEAD_FILL, size=9, font=F)
            ws.cell(row=hr1, column=c).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for j, sub in enumerate(subs):
            if not g:
                cellf(ws, hr1, c + j, None, fill=HEAD_FILL, size=9, font=F)
            cellf(ws, hr2, c + j, sub, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
            ws.cell(row=hr2, column=c + j).border = Border(bottom=THIN)
        c += len(subs)
    ncols = c - 1
    ws.row_dimensions[hr2].height = 54
    ws.row_dimensions[hr1].height = 28

    def blk(rec):
        k = rec["k"]
        return (k.dataset, k.backbone, "crop-only" if k.policy_id.startswith("croponly") else "screen")

    order = {("octmnist", "ResNet-50", "crop-only"): 0, ("octmnist", "ResNet-50", "screen"): 1,
             ("octmnist", "ResNet-18", "screen"): 2, ("pathmnist", "ResNet-50", "screen"): 3}

    def sortkey(rec):
        th = rec["probe"].get("Probe h, test")
        vh = rec["probe"].get("Probe h, validation")
        return (order.get(blk(rec), 9), -len(rec["ahead"]), -(th[2] if th else -99), -(vh[2] if vh else -99))

    rows.sort(key=sortkey)
    r, prev, out_csv = hr2 + 1, None, []
    for rec in rows:
        k = rec["k"]
        b = blk(rec)
        if b != prev:
            title = (f"{DNAME[k.dataset]}, {k.backbone}" + (": crop-only design" if b[2] == "crop-only"
                     else (": old-stem screen" if k.dataset == "octmnist" and k.backbone == "ResNet-50" else "")))
            for cc in range(1, ncols + 1):
                cellf(ws, r, cc, fill=BLOCK_FILL, size=9, font=F)
            cellf(ws, r, 1, title, bold=True, fill=BLOCK_FILL, size=9, font=F)
            r += 1; prev = b
        aug_pid = {"A0": "stem", "A6": "speckle_contrast_crop020", "croponly_stem": "stem"}.get(k.policy_id, k.policy_id)
        vals = [(DNAME[k.dataset], None), (k.backbone, None), (policy_info(k.policy_id, k.dataset)[0], None),
                (augs.get((DNAME[k.dataset], k.backbone, aug_pid)), None),
                (f"{rec['nv'][0]} / {rec['nv'][1]}", None), (f"{rec['nt'][0]} / {rec['nt'][1]}" if rec["nt"][0] else "0 / 0 (not in test plan)", None),
                (", ".join(rec["ahead"]), None)]
        fills = {}
        for lab, *_ in probes:
            p = rec["probe"].get(lab)
            if p is None:
                vals += [(None, None)] * 6
                continue
            (am, asd, _an), (bm, bsd, _bn), d, z = p
            col0 = len(vals) + 1
            vals += [(am, "0.00"), (asd, "0.00"), (bm, "0.00"), (bsd, "0.00"), (d, "+0.00;-0.00"), (z, "+0.0;-0.0")]
            strong = z is not None and abs(z) > 2
            if d != 0:
                fills[col0 + 4] = CMP_FILL[("up" if d > 0 else "down", strong)]
        vals += [(rec["sc"][0], "0.0"), (rec["sc"][1], "0.0"), (rec["scab"][0], "0.0"), (rec["scab"][1], "0.0"),
                 (rec["msv"][0], "0.000"), (rec["msv"][1], "0.000"),
                 (rec["sct"][0], "0.0"), (rec["sct"][1], "0.0")]
        (pa, pb), (ta, tb) = rec["pu"], rec["put"]
        pd_ = pa[0] - pb[0] if pa[0] is not None and pb[0] is not None else None
        td_ = ta[0] - tb[0] if ta[0] is not None and tb[0] is not None else None
        vals += [(pa[0], "0.000"), (pa[1], "0.000"), (pb[0], "0.000"), (pb[1], "0.000"), (pd_, "+0.000;-0.000"),
                 (ta[0], "0.000"), (tb[0], "0.000"), (td_, "+0.000;-0.000")]
        for j, (v, fmt) in enumerate(vals, start=1):
            fc = fills.get(j)
            cellf(ws, r, j, v, fmt=fmt, size=9, font=F, fill=PatternFill("solid", start_color=fc) if fc else None,
                  wrap=(j in (3, 4)))
        out_csv.append({h: v for h, (v, _f) in zip([f"{g} | {sub}" if g else sub for g, subs in groups for sub in subs], vals)})
        r += 1
    widths = [9, 10, 34, 46, 11, 11, 16] + [8.5] * (ncols - 7)
    for cc, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(cc)].width = w
    ws.freeze_panes = ws.cell(row=hr2 + 1, column=4)
    pd.DataFrame(out_csv).to_csv(OUT.parent / "ncl_ahead.csv", index=False)
    return len(rows)


def best_val_sheet(wb, ok, tdf):
    """Sheet '0c Best on validation': for each dataset, backbone and method, the run (one seed) and the policy (mean over
    seeds) with the highest validation linear probe, selected separately by the backbone probe (h) and the projector
    probe (z), with that choice's validation and test results. Selection uses validation only; test is read off."""
    ws = wb.create_sheet("0c Best on validation", 3)
    F = FULL_FONT
    augs = read_policy_augs()
    t = tdf.copy() if len(tdf) else tdf
    tcols = {"probe_bb_bal_acc": "t_h", "probe_proj_bal_acc": "t_z", "probe_bb_auc": "t_auc", "sc_pct": "t_sc",
             "purity_proj_star": "t_pu", "plan": "t_plan"}
    tsel = t[["run_id"] + [c for c in tcols if c in t.columns]].rename(columns=tcols) if len(t) else pd.DataFrame(columns=["run_id"])
    runs = ok.merge(tsel, on="run_id", how="left")
    for c in ("t_h", "t_z", "t_auc", "t_sc", "t_pu"):
        if c not in runs.columns:
            runs[c] = float("nan")
    cellf(ws, 1, 1, "Best on validation: the run and the policy with the highest validation linear probe, and their test results",
          bold=True, size=13, font=F)
    cellf(ws, 2, 1, "Chosen on validation only, separately by the backbone probe (h) and by the projector probe (z), for each dataset, "
                    "backbone and method; test is read off for the chosen run or policy. A single best run is the luckiest of its "
                    "seeds, so its numbers run high; the best policy (mean over seeds) is the fairer choice. MS is raw, with its chance "
                    "floor beside it. SimCLR's SC sits at chance by construction. Collapsed and diverged runs are left out.",
          size=9, italic=True, font=F, color=MUTED)
    head = ["Dataset", "Backbone", "Chosen by", "Method", "Policy", "Augmentations applied (level, probability per view)", "Seed(s)",
            "Val probe h", "Val probe z", "Val k-NN-20", "Test probe h", "Test probe z", "Test AUC h",
            "Val SC %", "Val SC above shuffle", "Val purity z*", "Val purity h*", "Val MS", "MS floor (chance)",
            "Test SC %", "Test purity z*", "Test plan"]
    fmts = [None] * 7 + ["0.00"] * 6 + ["0.0", "0.0", "0.000", "0.000", "0.0000", "0.0000", "0.0", "0.000", None]
    r = 4
    for block, title in (("run", "Best single run (one seed)"), ("policy", "Best policy (mean over seeds; policies with at least 3 seeds; SD in the note on each probe cell)")):
        for c in range(1, len(head) + 1):
            cellf(ws, r, c, fill=BLOCK_FILL, size=9, font=F)
        cellf(ws, r, 1, title, bold=True, fill=BLOCK_FILL, size=9, font=F)
        r += 1
        for j, h in enumerate(head, start=1):
            cellf(ws, r, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
            ws.cell(row=r, column=j).border = Border(bottom=THIN)
        ws.row_dimensions[r].height = 40
        r += 1
        for (ds, bb), g in runs.groupby(["dataset", "backbone"], sort=False):
            for crit, col in (("probe h", "probe_bb_bal_acc"), ("probe z", "probe_proj_bal_acc")):
                picked = {}
                for o, oname in OBJ:
                    go = g[g.objective == o]
                    if not len(go):
                        continue
                    if block == "run":
                        b = go.loc[go[col].idxmax()]
                        pid, seeds = b.policy_id, str(int(b.seed))
                        vals = {k: b.get(k) for k in ("probe_bb_bal_acc", "probe_proj_bal_acc", "knn20_bal_acc", "t_h", "t_z", "t_auc",
                                                      "sc_pct", "sc_above", "purity_proj_star", "purity_bb_star", "ms", "ms_floor",
                                                      "t_sc", "t_pu")}
                        sds, plan = {}, b.get("t_plan")
                    else:
                        cnt = go.groupby("policy_id")[col].count()
                        m = go.groupby("policy_id")[col].mean()[cnt >= 3]
                        if not len(m):
                            continue
                        pid = m.idxmax()
                        gp = go[go.policy_id == pid]
                        seeds = ", ".join(str(int(x)) for x in sorted(gp.seed))
                        num = lambda k: pd.to_numeric(gp[k], errors="coerce") if k in gp.columns else pd.Series(dtype=float)
                        vals = {k: (float(num(k).mean()) if num(k).notna().any() else None)
                                for k in ("probe_bb_bal_acc", "probe_proj_bal_acc", "knn20_bal_acc", "t_h", "t_z", "t_auc", "sc_pct",
                                          "sc_above", "purity_proj_star", "purity_bb_star", "ms", "ms_floor", "t_sc", "t_pu")}
                        sds = {k: (float(num(k).std(ddof=1)) if num(k).notna().sum() > 1 else None)
                               for k in ("probe_bb_bal_acc", "probe_proj_bal_acc", "t_h", "t_z")}
                        plans = sorted({PLAN_LABEL.get(str(x).split("/")[-1], str(x)) for x in gp.get("t_plan", pd.Series()).dropna()})
                        plan = "; ".join(plans) if plans else None
                    picked[o] = vals
                    aug_pid = {"A0": "stem", "A6": "speckle_contrast_crop020"}.get(pid, pid)
                    row = [DNAME[ds], bb, crit, oname, policy_info(pid, ds)[0], augs.get((DNAME[ds], bb, aug_pid)), seeds,
                           vals["probe_bb_bal_acc"], vals["probe_proj_bal_acc"], vals["knn20_bal_acc"], vals["t_h"], vals["t_z"],
                           vals["t_auc"], vals["sc_pct"], vals["sc_above"] if o == "ncl" else None, vals["purity_proj_star"],
                           vals["purity_bb_star"], vals["ms"], vals["ms_floor"], vals["t_sc"], vals["t_pu"],
                           PLAN_LABEL.get(str(plan).split("/")[-1], plan) if block == "run" and plan is not None else plan]
                    for j, (v, fm) in enumerate(zip(row, fmts), start=1):
                        v = None if (isinstance(v, float) and math.isnan(v)) else v
                        hit = (j == 8 and col == "probe_bb_bal_acc") or (j == 9 and col == "probe_proj_bal_acc")
                        cellf(ws, r, j, v, fmt=fm, bold=hit, size=9, font=F, wrap=(j == 6))
                    if block == "policy":
                        for j, k in ((8, "probe_bb_bal_acc"), (9, "probe_proj_bal_acc"), (11, "t_h"), (12, "t_z")):
                            if sds.get(k) is not None:
                                ws.cell(row=r, column=j).comment = Comment(f"SD over seeds: {sds[k]:.2f}", "make_results_workbook.py",
                                                                           width=160, height=40)
                    r += 1
                if "ncl" in picked and "simclr" in picked:
                    cellf(ws, r, 4, "NCL − SimCLR", italic=True, size=9, font=F, color=MUTED)
                    for j, k in ((8, "probe_bb_bal_acc"), (9, "probe_proj_bal_acc"), (10, "knn20_bal_acc"), (11, "t_h"), (12, "t_z"),
                                 (13, "t_auc"), (16, "purity_proj_star"), (17, "purity_bb_star"), (21, "t_pu")):
                        a, b2 = picked["ncl"].get(k), picked["simclr"].get(k)
                        if a is None or b2 is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b2, float) and math.isnan(b2)):
                            continue
                        d = float(a) - float(b2)
                        fc = CMP_FILL[("up" if d > 0 else "down", False)] if d != 0 else None
                        cellf(ws, r, j, d, fmt="+0.000;-0.000" if j in (16, 17, 21) else "+0.00;-0.00", italic=True, size=9, font=F,
                              fill=PatternFill("solid", start_color=fc) if fc else None)
                    r += 1
            r += 0
        r += 1
    widths = [9, 9, 8, 7, 30, 44, 9] + [8] * (len(head) - 8) + [18]
    for c, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(c)].width = w

# ---------------------------------------------------------------- summary sheets (policy means, sweeps, best seeds)
FAMILIES = [  # (family, block, [(level label, policy id), ...]); the first level is the family's base
    ("Brightness strength (p 0.8)", "crop-only base", [("off", "croponly"), ("0.2", "croponly_bright_b020"), ("0.4", "croponly_bright_b040"),
                                                      ("0.8", "croponly_bright"), ("0.95", "croponly_bright_b095")]),
    ("Brightness × flip", "crop-only base", [("neither (crop only)", "croponly"), ("brightness 0.8", "croponly_bright"),
                                              ("flip", "croponly_flip"), ("both (= stem)", "stem")]),
    ("Contrast factor (p 0.5)", "crop-only base", [("off", "croponly"), ("0.8", "croponly_contrast"), ("0.5", "croponly_contrast_f050"),
                                                  ("0.3", "croponly_contrast_f030"), ("0.1", "croponly_contrast_f010")]),
    ("Blur σ max (p 0.5)", "crop-only base", [("off", "croponly"), ("0.6", "croponly_blur_sig060"), ("1.2", "croponly_blur"),
                                               ("2.0", "croponly_blur_sig200")]),
    ("Minimum crop area", "crop-only base", [("0.08", "croponly"), ("0.2", "croponly_crop020"), ("0.5", "croponly_crop050")]),
    ("Speckle std (p 0.5)", "crop-only base", [("off", "croponly"), ("0.05", "croponly_speckle_s005"), ("0.1", "croponly_speckle"),
                                              ("0.2", "croponly_speckle_s020")]),
    ("Motion radius, sigma (p 0.5)", "crop-only base", [("off", "croponly"), ("3, 1", "croponly_motion_r3"), ("5, 2", "croponly_motion"),
                                                       ("8, 3", "croponly_motion_r8")]),
    ("Pixelation factor (p 0.5)", "crop-only base", [("off", "croponly"), ("0.9", "croponly_pixelate_f090"), ("0.8", "croponly_pixelate"),
                                                    ("0.6", "croponly_pixelate_f060")]),
    ("JPEG quality (p 0.5)", "crop-only base", [("off", "croponly"), ("90", "croponly_jpeg_q90"), ("70", "croponly_jpeg"),
                                               ("40", "croponly_jpeg_q40")]),
    ("Projector output size", "crop-only base", [("256", "croponly"), ("2048", "croponly_proj2048")]),
    ("Contrast factor (p 0.5)", "old stem", [("off (stem)", "stem"), ("0.9", "contrast_f090"), ("0.8", "contrast"), ("0.6", "contrast_f060")]),
    ("Minimum crop area", "old stem", [("0.08 (stem)", "stem"), ("0.2", "stem_crop020"), ("0.5", "stem_crop050")]),
    ("Minimum crop area", "old stem + contrast", [("0.08", "contrast"), ("0.2", "contrast_crop020"), ("0.5", "contrast_crop050")]),
]
METRICS = {"val_h": ("v", "probe_bb_bal_acc"), "val_z": ("v", "probe_proj_bal_acc"), "knn": ("v", "knn20_bal_acc"),
           "test_h": ("t", "probe_bb_bal_acc"), "test_z": ("t", "probe_proj_bal_acc"), "sc": ("v", "sc_pct"), "scab": ("v", "sc_above"),
           "ms": ("v", "ms"), "floor": ("v", "ms_floor"), "pu_z": ("v", "purity_proj_star"), "pu_h": ("v", "purity_bb_star"),
           "ed_h": ("v", "effdim_bb"), "ed_z": ("v", "effdim_proj"), "t_sc": ("t", "sc_pct"), "t_pu": ("t", "purity_proj_star"),
           "t_scab": ("t", "sc_above"), "dead": ("v", "dead_proj"), "ms_star": ("v", "ms_star"), "t_knn": ("t", "knn20_bal_acc"),
           "pu_z_raw": ("v", "purity_proj"), "pu_h_raw": ("v", "purity_bb"),
           "t_ms": ("t", "test_ms"), "t_ms_star": ("t", "test_ms_star")}
STAT_COLS = [("Val probe h mean", "val_h", 0, "0.00"), ("Val probe h SD", "val_h", 1, "0.00"),
             ("Val probe z mean", "val_z", 0, "0.00"), ("Val probe z SD", "val_z", 1, "0.00"),
             ("Val k-NN-20 mean", "knn", 0, "0.00"), ("Val k-NN-20 SD", "knn", 1, "0.00"),
             ("Test probe h mean", "test_h", 0, "0.00"), ("Test probe h SD", "test_h", 1, "0.00"),
             ("Test probe z mean", "test_z", 0, "0.00"), ("Test probe z SD", "test_z", 1, "0.00"), ("Test k-NN-20 mean", "t_knn", 0, "0.00"),
             ("Val SC % mean", "sc", 0, "0.0"), ("Val SC % SD", "sc", 1, "0.0"),
             ("Val SC above shuffle mean", "scab", 0, "0.0"), ("Val SC above shuffle SD", "scab", 1, "0.0"),
             ("Val MS mean", "ms", 0, "0.0000"), ("Val MS SD", "ms", 1, "0.0000"),
             ("Test MS mean", "t_ms", 0, "0.0000"), ("Test MS SD", "t_ms", 1, "0.0000"),
             ("Val purity z mean", "pu_z_raw", 0, "0.000"), ("Val purity z SD", "pu_z_raw", 1, "0.000"),
             ("Val purity h mean", "pu_h_raw", 0, "0.000"),
             ("Eff. dim h mean", "ed_h", 0, "0.0"), ("Eff. dim z mean", "ed_z", 0, "0.0"),
             ("Test SC % mean", "t_sc", 0, "0.0"), ("Test SC above shuffle mean", "t_scab", 0, "0.0"), ("Test purity z* mean", "t_pu", 0, "0.000")]
UTIL_COLS = [("Val probe h mean", "val_h", 0, "0.00"), ("Val probe z mean", "val_z", 0, "0.00"),
             ("Val SC % mean", "sc", 0, "0.0"), ("Val SC % SD", "sc", 1, "0.0"),
             ("Val SC above shuffle mean", "scab", 0, "0.0"), ("Val SC above shuffle SD", "scab", 1, "0.0"),
             ("Val MS mean", "ms", 0, "0.0000"), ("Val MS SD", "ms", 1, "0.0000"), ("Val MS* mean", "ms_star", 0, "0.000"),
             ("Test MS mean", "t_ms", 0, "0.0000"), ("Test MS* mean", "t_ms_star", 0, "0.000"),
             ("Val purity z (raw) mean", "pu_z_raw", 0, "0.000"), ("Val purity z* mean", "pu_z", 0, "0.000"),
             ("Val purity z* SD", "pu_z", 1, "0.000"), ("Val purity h (raw) mean", "pu_h_raw", 0, "0.000"),
             ("Val purity h* mean", "pu_h", 0, "0.000"), ("Val purity h* SD", "pu_h", 1, "0.000"),
             ("Eff. dim h mean", "ed_h", 0, "0.0"), ("Eff. dim h SD", "ed_h", 1, "0.0"),
             ("Eff. dim z mean", "ed_z", 0, "0.0"), ("Eff. dim z SD", "ed_z", 1, "0.0"),
             ("Dead projector units mean", "dead", 0, "0.0"),
             ("Test SC above shuffle mean", "t_scab", 0, "0.0"), ("Test purity z* mean", "t_pu", 0, "0.000")]
DIFF_KEYS = {"val_h", "val_z", "knn", "test_h", "test_z", "t_knn", "ms", "ms_star", "pu_z", "pu_h", "pu_z_raw", "pu_h_raw",
             "ed_h", "ed_z", "t_pu", "dead", "t_ms", "t_ms_star"}
RUN_RE = re.compile(r"^(ncl|simclr)-(resnet\d+)-(octmnist|pathmnist)-(.+?)(?:-s(\d+))?$")


def read_diverged():
    """Run ids whose features came out non-finite (diverged training), from the scorers' lists."""
    out = set()
    for p in (Path("study/nonfinite_runs.txt"), Path("study/nonfinite_runs_test.txt"), Path("study/path/nonfinite_runs.txt")):
        if p.exists():
            out |= {l.split("\t")[0].strip() for l in p.read_text().splitlines() if l.strip()}
    return out


class PolicyStats:
    def __init__(self, val_all, ok, tdf):
        self.val_all, self.ok, self.tdf = val_all, ok, tdf
        self.div = {}
        for rid in read_diverged():
            m = RUN_RE.match(rid)
            if m and rid not in set(ok.run_id):
                bb = {"resnet50": "ResNet-50", "resnet18": "ResNet-18"}.get(m.group(2), m.group(2))
                key = (m.group(3), bb, m.group(4), m.group(1))
                self.div[key] = self.div.get(key, 0) + 1
        self.sd_cache = {}

    def sel(self, df, ds, bb, pid, o):
        if not len(df):
            return df
        return df[(df.dataset == ds) & (df.backbone == bb) & (df.policy_id == pid) & (df.objective == o)]

    def get(self, ds, bb, pid, o):
        pid = {"croponly_stem": "stem"}.get(pid, pid)
        v, t, a = self.sel(self.ok, ds, bb, pid, o), self.sel(self.tdf, ds, bb, pid, o), self.sel(self.val_all, ds, bb, pid, o)
        out = {"n_val": len(v), "n_test": len(t), "pid": pid, "collapsed": int((pd.to_numeric(a.probe_bb_bal_acc, errors="coerce") < COLLAPSE_BELOW).sum()) if len(a) else 0,
               "diverged": self.div.get((ds, bb, pid, o), 0),
               "seeds": ", ".join(str(int(x)) for x in sorted(pd.to_numeric(v.seed, errors="coerce").dropna())) if len(v) else ""}
        for k, (src, col) in METRICS.items():
            d = v if src == "v" else t
            x = pd.to_numeric(d[col], errors="coerce").dropna() if len(d) and col in d.columns else pd.Series(dtype=float)
            out[k] = (float(x.mean()) if len(x) else None, float(x.std(ddof=1)) if len(x) > 1 else None, len(x))
        return out

    def se(self, ds, bb, key, na, nb):
        src, col = METRICS[key]
        df = self.ok if src == "v" else self.tdf
        sds = []
        for o in ("ncl", "simclr"):
            ck = (src, ds, bb, o, col)
            if ck not in self.sd_cache:
                self.sd_cache[ck] = pooled_sd(df, ds, bb, o, col)
            sds.append(self.sd_cache[ck])
        if None in sds or not na or not nb:
            return None
        return math.sqrt(sds[0] ** 2 / na + sds[1] ** 2 / nb)


def _write_stat_block(ws, r, ident, stn, sts, ds, bb, PS, extra=None, F=FULL_FONT, cols=None):
    """Rows NCL, SimCLR and NCL - SimCLR for one policy. ident = identity values shared by both methods."""
    cols = cols or STAT_COLS
    for j, (st, oname) in enumerate(((stn, "NCL"), (sts, "SimCLR"))):
        attempted = st["n_val"] + st["collapsed"] + st["diverged"]
        row = list(ident) + [oname, st["seeds"], attempted, st["n_val"], st["collapsed"], st["diverged"], st["n_test"]]
        if extra:
            row += extra[j]
        row += [st[k][i] if st[k][0] is not None else None for _h, k, i, _f in cols]
        fmts = [None] * (len(ident) + 7) + (list(extra[2]) if extra else []) + [f for *_x, f in cols]
        for c, (v, fm) in enumerate(zip(row, fmts), start=1):
            cellf(ws, r, c, v, fmt=fm, size=9, font=F)
        r += 1
    cellf(ws, r, len(ident) + 1, "NCL − SimCLR", italic=True, size=9, font=F, color=MUTED)
    c0 = len(ident) + 7 + (len(extra[0]) if extra else 0)
    for c, (_h, k, i, fm) in enumerate(cols, start=c0 + 1):
        if i != 0 or k not in DIFF_KEYS or stn[k][0] is None or sts[k][0] is None:
            continue
        d = stn[k][0] - sts[k][0]
        se = PS.se(ds, bb, k, stn[k][2], sts[k][2])
        strong = se is not None and se > 0 and abs(d) > 2 * se
        fc = CMP_FILL[("up" if d > 0 else "down", strong)] if d != 0 else None
        cellf(ws, r, c, d, fmt=("+" + fm + ";-" + fm), italic=True, size=9, font=F, fill=PatternFill("solid", start_color=fc) if fc else None)
    return r + 1


def _stat_header(ws, r, ident_heads, extra_heads=(), F=FULL_FONT, cols=None):
    cols = cols or STAT_COLS
    heads = (list(ident_heads) + ["Method", "Seeds (valid)", "Attempted", "n val (valid)", "Collapsed", "Diverged", "n test"]
             + list(extra_heads) + [h for h, *_x in cols])
    for j, h in enumerate(heads, start=1):
        cellf(ws, r, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
        ws.cell(row=r, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[r].height = 42
    return heads


def all_policy_means_sheet(wb, PS, keys, augs, name="3 Matched method comparison", cols=None,
                           title="Matched method comparison: NCL and SimCLR under the same policy, means and SDs over seeds"):
    ws = wb.create_sheet(name)
    F = FULL_FONT
    cellf(ws, 1, 1, title, bold=True, size=13, font=F)
    cellf(ws, 2, 1, "One block per policy: the NCL row, the SimCLR row and NCL − SimCLR (green: NCL higher, pink: lower; darker = beyond "
                    "2 SE, seed SD pooled over policies). Collapsed (probe below 60) and diverged (non-finite) runs are counted and left "
                    "out of the means. SC % and SC above shuffle are NCL measures (SimCLR's SC is at chance by construction); MS is raw, "
                    "with the dataset's chance floor given in '9 Methods and notes'; MS is scored on validation only. Means are over "
                    "successful runs (n val); Attempted counts every run trained.", size=9, italic=True, font=F, color=MUTED)
    heads = _stat_header(ws, 4, ["Dataset", "Backbone", "Policy", "Policy id", "Augmentations applied"], cols=cols)
    r, prev = 5, None
    for k in keys.itertuples():
        blk = (k.dataset, k.backbone, "crop-only" if k.policy_id.startswith("croponly") else "screen")
        if blk != prev:
            for c in range(1, len(heads) + 1):
                cellf(ws, r, c, fill=BLOCK_FILL, size=9, font=F)
            cellf(ws, r, 1, f"{DNAME[k.dataset]}, {k.backbone}" + (": crop-only design" if blk[2] == "crop-only" else ""),
                  bold=True, fill=BLOCK_FILL, size=9, font=F)
            r += 1; prev = blk
        aug_pid = {"A0": "stem", "A6": "speckle_contrast_crop020", "croponly_stem": "stem"}.get(k.policy_id, k.policy_id)
        ident = [DNAME[k.dataset], k.backbone, policy_info(k.policy_id, k.dataset)[0], k.policy_id,
                 augs.get((DNAME[k.dataset], k.backbone, aug_pid))]
        r = _write_stat_block(ws, r, ident, PS.get(k.dataset, k.backbone, k.policy_id, "ncl"),
                              PS.get(k.dataset, k.backbone, k.policy_id, "simclr"), k.dataset, k.backbone, PS, cols=cols)
    for c, w in enumerate([9, 9, 30, 22, 40] + [7] * (len(heads) - 5), start=1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = ws.cell(row=5, column=4)


def sweeps_sheet(wb, PS, augs):
    ws = wb.create_sheet("5a Sweeps detail")
    F = FULL_FONT
    ds, bb = "octmnist", "ResNet-50"
    cellf(ws, 1, 1, "Augmentation sweeps (OCT, ResNet-50): one augmentation varied at a time", bold=True, size=13, font=F)
    cellf(ws, 2, 1, "Each family starts from its base level (first row). '− base' columns are each method's change from that base. "
                    "Brightness × flip is the full 2×2 on the crop-only base (the stem is its 'both' cell). Levels without "
                    "results yet are listed as such. Rows NCL, SimCLR and NCL − SimCLR as in '3 Matched method comparison'.",
          size=9, italic=True, font=F, color=MUTED)
    extra_heads = ["Val probe h − base", "Test probe h − base", "SC above shuffle − base"]
    heads = _stat_header(ws, 4, ["Family", "Base", "Level", "Policy id", "Augmentations applied"], extra_heads)
    r = 5
    for fam, base_name, levels in FAMILIES:
        if fam.startswith("Brightness \u00d7"):
            continue
        for c in range(1, len(heads) + 1):
            cellf(ws, r, c, fill=BLOCK_FILL, size=9, font=F)
        cellf(ws, r, 1, f"{fam} — {base_name}", bold=True, fill=BLOCK_FILL, size=9, font=F)
        r += 1
        base = {o: PS.get(ds, bb, levels[0][1], o) for o in ("ncl", "simclr")}
        for lab, pid in levels:
            stn, sts = PS.get(ds, bb, pid, "ncl"), PS.get(ds, bb, pid, "simclr")
            ident = [fam, base_name, lab, pid, augs.get(("OCT", bb, pid))]
            if stn["n_val"] == 0 and sts["n_val"] == 0 and stn["n_test"] == 0 and sts["n_test"] == 0:
                for j, v in enumerate(ident + ["no results yet" if not (stn["diverged"] or sts["diverged"]) else "diverged"], start=1):
                    cellf(ws, r, j, v, size=9, font=F, italic=(j == len(ident) + 1), color=MUTED if j == len(ident) + 1 else None)
                r += 1
                continue
            def dlt(st, o):
                g = lambda kk: (st[kk][0], base[o][kk][0])
                return [None if a is None or b is None else a - b for a, b in (g("val_h"), g("test_h"), g("scab"))]
            extra = [dlt(stn, "ncl"), dlt(sts, "simclr"), ["+0.00;-0.00", "+0.00;-0.00", "+0.0;-0.0"]]
            if fam.startswith("Projector"):
                ident[1] = base_name + " (architecture, a separate factor)"
            r = _write_stat_block(ws, r, ident, stn, sts, ds, bb, PS, extra=extra)
        r += 1
    for c, w in enumerate([22, 14, 14, 22, 34] + [7] * (len(heads) - 5), start=1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = ws.cell(row=5, column=4)


def sweep_best_sheet(wb, PS):
    """For each sweep level and method, the seed with the best validation probe (h, z) and that same run's other results."""
    ws = wb.create_sheet("5b Sweep best seeds")
    F = FULL_FONT
    ds, bb = "octmnist", "ResNet-50"
    cellf(ws, 1, 1, "Sweep levels: the seed with the best validation probe, and that run's results", bold=True, size=13, font=F)
    cellf(ws, 2, 1, "Chosen on validation only; test is the same run's. The best of a few seeds runs high by luck: use the policy means "
                    "in '5 Severity sweeps' for comparisons.", size=9, italic=True, font=F, color=MUTED)
    heads = ["Family", "Base", "Level", "Policy id", "Method", "Chosen by", "Seeds available", "Best seed", "Val probe h", "Val probe z",
             "Test probe h (same run)", "Test probe z (same run)", "Val SC %", "Val SC above shuffle", "Val MS", "Val purity z", "Run id"]
    fm = [None] * 8 + ["0.00"] * 4 + ["0.0", "0.0", "0.0000", "0.000", None]
    for j, h in enumerate(heads, start=1):
        cellf(ws, 4, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
        ws.cell(row=4, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[4].height = 42
    tt = PS.tdf.set_index("run_id") if len(PS.tdf) else pd.DataFrame()
    r = 5
    for fam, base_name, levels in FAMILIES:
        for lab, pid in levels:
            for o, oname in OBJ:
                v = PS.sel(PS.ok, ds, bb, pid, o)
                if not len(v):
                    continue
                for crit, col in (("probe h", "probe_bb_bal_acc"), ("probe z", "probe_proj_bal_acc")):
                    b = v.loc[pd.to_numeric(v[col], errors="coerce").idxmax()]
                    th = tt.loc[b.run_id, "probe_bb_bal_acc"] if b.run_id in tt.index else None
                    tz = tt.loc[b.run_id, "probe_proj_bal_acc"] if b.run_id in tt.index else None
                    row = [fam, base_name, lab, pid, oname, crit, len(v), int(b.seed), num(b.probe_bb_bal_acc), num(b.probe_proj_bal_acc),
                           num(th), num(tz), num(b.get("sc_pct")), num(b.get("sc_above")), num(b.get("ms")), num(b.get("purity_proj")), b.run_id]
                    for j, (val_, f_) in enumerate(zip(row, fm), start=1):
                        cellf(ws, r, j, val_, fmt=f_, size=9, font=F, bold=(j == 9 and crit == "probe h") or (j == 10 and crit == "probe z"))
                    r += 1
    for c, w in enumerate([22, 14, 14, 22, 7, 8, 8, 6] + [8] * 8 + [48], start=1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = ws.cell(row=5, column=5)


def ref_policy(pid):
    if pid in ("stem", "croponly"):
        return None
    return "croponly" if pid.startswith("croponly") else "stem"


EFFECT_METRICS = [("Val probe h", "val_h", "+0.00;-0.00"), ("Val probe z", "val_z", "+0.00;-0.00"), ("Val k-NN-20", "knn", "+0.00;-0.00"),
                  ("Test probe h", "test_h", "+0.00;-0.00"), ("Test probe z", "test_z", "+0.00;-0.00"),
                  ("Val SC above shuffle", "scab", "+0.0;-0.0"), ("Val MS", "ms", "+0.0000;-0.0000"),
                  ("Val purity z*", "pu_z", "+0.000;-0.000"), ("Val purity h*", "pu_h", "+0.000;-0.000"),
                  ("Eff. dim h", "ed_h", "+0.0;-0.0"), ("Eff. dim z", "ed_z", "+0.0;-0.0")]


def _pooled(PS, ds, bb, o, key):
    src, col = METRICS[key]
    ck = ("one", src, ds, bb, o, col)
    if ck not in PS.sd_cache:
        PS.sd_cache[ck] = pooled_sd(PS.ok if src == "v" else PS.tdf, ds, bb, o, col)
    return PS.sd_cache[ck]


def _fill_z(d, se):
    if d is None or d == 0:
        return None
    strong = se is not None and se > 0 and abs(d) > 2 * se
    return PatternFill("solid", start_color=CMP_FILL[("up" if d > 0 else "down", strong)])


def augmentation_effects_sheet(wb, PS, keys, augs):
    """Each policy's change from its reference (stem, or crop only), per method, and the NCL - SimCLR difference of changes."""
    ws = wb.create_sheet("4 Augmentation effects")
    F = FULL_FONT
    cellf(ws, 1, 1, "Augmentation effects: each policy's change from its reference, for NCL and SimCLR", bold=True, size=13, font=F)
    cellf(ws, 2, 1, "Change = policy mean minus reference mean (reference: the stem, or crop only for the crop-only design). "
                    "'NCL − SimCLR' = NCL's change minus SimCLR's change (does the augmentation act differently under non-negativity). "
                    "Green: increase, pink: decrease; darker = beyond 2 SE (seed SD pooled over policies, per method). One-seed policies "
                    "are a screen. SC above shuffle and MS are shown for NCL only (SimCLR sits at chance / at the floor).", size=9, italic=True, font=F, color=MUTED)
    ident_h = ["Dataset", "Backbone", "Block", "Policy", "Policy id", "Reference", "Augmentations applied", "Seeds NCL", "Seeds SimCLR"]
    heads = list(ident_h)
    for lab, _k, _f in EFFECT_METRICS:
        heads += [f"{lab}: NCL change", f"{lab}: SimCLR change", f"{lab}: NCL − SimCLR"]
    for j, h in enumerate(heads, start=1):
        cellf(ws, 4, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
        ws.cell(row=4, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[4].height = 54
    r = 5
    for k in keys.itertuples():
        ref = ref_policy(k.policy_id)
        if ref is None:
            continue
        ds, bb = k.dataset, k.backbone
        P = {o: PS.get(ds, bb, k.policy_id, o) for o in ("ncl", "simclr")}
        R = {o: PS.get(ds, bb, ref, o) for o in ("ncl", "simclr")}
        aug_pid = {"A0": "stem", "A6": "speckle_contrast_crop020", "croponly_stem": "stem"}.get(k.policy_id, k.policy_id)
        row = [DNAME[ds], bb, "crop-only design" if k.policy_id.startswith("croponly") else "old-stem screen",
               policy_info(k.policy_id, ds)[0], k.policy_id, {"croponly": "crop only", "stem": "stem"}[ref],
               augs.get((DNAME[ds], bb, aug_pid)), P["ncl"]["seeds"], P["simclr"]["seeds"]]
        for c, v in enumerate(row, start=1):
            cellf(ws, r, c, v, size=9, font=F, wrap=(c == 7))
        c = len(row) + 1
        for lab, key, fm in EFFECT_METRICS:
            ch, se = {}, {}
            for o in ("ncl", "simclr"):
                a, b = P[o][key], R[o][key]
                if a[0] is None or b[0] is None or (o == "simclr" and key in ("scab", "ms")):
                    ch[o] = se[o] = None
                    continue
                ch[o] = a[0] - b[0]
                sd = _pooled(PS, ds, bb, o, key)
                se[o] = sd * math.sqrt(1 / a[2] + 1 / b[2]) if sd is not None and a[2] and b[2] else None
            for j, o in enumerate(("ncl", "simclr")):
                cellf(ws, r, c + j, ch[o], fmt=fm, size=9, font=F, fill=_fill_z(ch[o], se[o]))
            if ch["ncl"] is not None and ch["simclr"] is not None:
                d = ch["ncl"] - ch["simclr"]
                s2 = math.sqrt(se["ncl"] ** 2 + se["simclr"] ** 2) if se["ncl"] is not None and se["simclr"] is not None else None
                cellf(ws, r, c + 2, d, fmt=fm, italic=True, size=9, font=F, fill=_fill_z(d, s2))
            c += 3
        r += 1
    for cc, w in enumerate([9, 9, 13, 30, 22, 9, 38, 9, 9] + [8] * (len(heads) - 9), start=1):
        ws.column_dimensions[get_column_letter(cc)].width = w
    ws.freeze_panes = ws.cell(row=5, column=6)


def brightness_flip_sheet(wb, PS):
    """The replicated 2x2 on the crop-only base: crop only, + brightness, + flip, + both (= stem)."""
    ws = wb.create_sheet("6 Brightness × flip")
    F = FULL_FONT
    ds, bb = "octmnist", "ResNet-50"
    cells = [("Crop only", "croponly"), ("+ brightness 0.8", "croponly_bright"), ("+ flip", "croponly_flip"), ("+ both (= stem)", "stem")]
    cellf(ws, 1, 1, "Brightness × flip (OCT, ResNet-50): the replicated 2×2 on the crop-only base", bold=True, size=13, font=F)
    cellf(ws, 2, 1, "Cell means over seeds. Brightness effect = (+ brightness) − (crop only); flip effect = (+ flip) − (crop only); "
                    "interaction = (both − flip) − (brightness − crop only), the extra change when both are on. SE from the "
                    "seed SD pooled over policies for that method; |z| > 2 is beyond noise (darker fill).", size=9, italic=True, font=F, color=MUTED)
    heads = ["Metric", "Method"] + [lab for lab, _p in cells] + ["Seeds per cell", "Brightness effect", "Flip effect", "Interaction",
                                                                 "Interaction SE", "Interaction z"]
    for j, h in enumerate(heads, start=1):
        cellf(ws, 4, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
        ws.cell(row=4, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[4].height = 40
    r = 5
    for lab, key, fm in EFFECT_METRICS:
        base_fm = fm.split(";")[0].lstrip("+")
        for o, oname in OBJ:
            if o == "simclr" and key in ("scab", "ms"):   # SimCLR sits at chance on these by construction / measurement
                continue
            st = [PS.get(ds, bb, pid, o)[key] for _l, pid in cells]
            if any(x[0] is None for x in st):
                continue
            m = [x[0] for x in st]
            n = [x[2] for x in st]
            be, fe, it = m[1] - m[0], m[2] - m[0], (m[3] - m[2]) - (m[1] - m[0])
            sd = _pooled(PS, ds, bb, o, key)
            se = sd * math.sqrt(sum(1 / x for x in n)) if sd is not None and all(n) else None
            z = it / se if se else None
            row = [lab, oname] + m + [" / ".join(str(x) for x in n), be, fe, it, se, z]
            fmts = [None, None] + [base_fm] * 4 + [None, fm, fm, fm, base_fm, "+0.0;-0.0"]
            for c, (v, f_) in enumerate(zip(row, fmts), start=1):
                cellf(ws, r, c, v, fmt=f_, size=9, font=F, fill=_fill_z(it, se) if c == 10 else None)
            r += 1
    for cc, w in enumerate([18, 8, 10, 12, 10, 12, 12, 11, 10, 11, 11, 10], start=1):
        ws.column_dimensions[get_column_letter(cc)].width = w


def best_method_sheet(wb, PS, o, name, augs):
    """Best validation checkpoints for ONE method, chosen independently by probe h and by probe z, with the same run's test
    results and its policy's means; then the best policy by mean (3 seeds or more)."""
    ws = wb.create_sheet(name)
    F = FULL_FONT
    oname = dict(OBJ)[o]
    cellf(ws, 1, 1, f"Best validation results for {oname}, chosen independently of the other method", bold=True, size=13, font=F)
    cellf(ws, 2, 1, "Chosen on validation only, by the backbone probe (h) and by the projector probe (z); test values belong to the same run "
                    "(or the same policy). The best of several seeds runs high by luck; the policy columns show what the policy does on "
                    "average. Collapsed and diverged runs cannot be chosen.", size=9, italic=True, font=F, color=MUTED)
    base_txt = lambda pid: ("crop-only base: crop 0.08, no brightness or flip" if pid.startswith("croponly")
                            else "SimCLR / NCL default recipe" if pid == "cifar"
                            else "old stem: crop 0.08 · brightness 0.8 (p 0.8) · flip (p 0.5)")
    heads = ["Dataset", "Backbone", "Selection", "Chosen by", "Policy", "Policy id", "Stem / base", "Augmentations applied", "Best seed",
             "Run id", "Val probe h", "Val probe z", "Val k-NN-20", "Test probe h", "Test probe z", "Test k-NN-20", "Test AUC h",
             "Val SC %", "Val SC above shuffle", "Val MS", "Val purity z", "Val purity h", "Eff. dim h", "Eff. dim z",
             "Test SC %", "Test SC above shuffle", "Test purity z*",
             "Policy: attempted", "Policy: valid seeds", "Policy: collapsed", "Policy: diverged",
             "Policy val h mean", "Policy val h SD", "Policy val z mean", "Policy val z SD",
             "Policy test h mean", "Policy test h SD", "Policy test z mean", "Policy test z SD"]
    fm = [None] * 10 + ["0.00"] * 7 + ["0.0", "0.0", "0.0000", "0.000", "0.000", "0.0", "0.0", "0.0", "0.0", "0.000"] + ["0"] * 4 + ["0.00"] * 8
    for j, h in enumerate(heads, start=1):
        cellf(ws, 4, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
        ws.cell(row=4, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[4].height = 42
    tt = PS.tdf.set_index("run_id") if len(PS.tdf) else pd.DataFrame()
    tv = lambda rid, col: num(tt.loc[rid, col]) if rid in tt.index and col in tt.columns else None
    r = 5
    runs = PS.ok[PS.ok.objective == o]
    for (ds, bb), g in runs.groupby(["dataset", "backbone"], sort=False):
        for crit, col in (("probe h", "probe_bb_bal_acc"), ("probe z", "probe_proj_bal_acc")):
            for sel in ("best run", "best policy (mean, 3+ seeds)"):
                if sel == "best run":
                    b = g.loc[pd.to_numeric(g[col], errors="coerce").idxmax()]
                    pid, rid, seed = b.policy_id, b.run_id, int(b.seed)
                    runvals = [num(b.probe_bb_bal_acc), num(b.probe_proj_bal_acc), num(b.knn20_bal_acc), tv(rid, "probe_bb_bal_acc"),
                               tv(rid, "probe_proj_bal_acc"), tv(rid, "knn20_bal_acc"), tv(rid, "probe_bb_auc"), num(b.get("sc_pct")),
                               num(b.get("sc_above")), num(b.get("ms")), num(b.get("purity_proj")), num(b.get("purity_bb")),
                               num(b.get("effdim_bb")), num(b.get("effdim_proj")), tv(rid, "sc_pct"), tv(rid, "sc_above"), tv(rid, "purity_proj_star")]
                else:
                    cnt = g.groupby("policy_id")[col].count()
                    mm = g.groupby("policy_id")[col].mean()[cnt >= 3]
                    if not len(mm):
                        continue
                    pid, rid, seed = mm.idxmax(), None, None
                    st = PS.get(ds, bb, pid, o)
                    runvals = [st[k][0] for k in ("val_h", "val_z", "knn", "test_h", "test_z", "t_knn")] + [None] + \
                              [st[k][0] for k in ("sc", "scab", "ms", "pu_z_raw", "pu_h_raw", "ed_h", "ed_z", "t_sc", "t_scab", "t_pu")]
                st = PS.get(ds, bb, pid, o)
                aug_pid = {"A0": "stem", "A6": "speckle_contrast_crop020"}.get(pid, pid)
                row = [DNAME[ds], bb, sel, crit, policy_info(pid, ds)[0], pid, base_txt(pid), augs.get((DNAME[ds], bb, aug_pid)), seed, rid] + runvals
                row += [st["n_val"] + st["collapsed"] + st["diverged"], st["n_val"], st["collapsed"], st["diverged"],
                        st["val_h"][0], st["val_h"][1], st["val_z"][0], st["val_z"][1], st["test_h"][0], st["test_h"][1], st["test_z"][0], st["test_z"][1]]
                for c, (v, f_) in enumerate(zip(row, fm), start=1):
                    hit = (c == 11 and crit == "probe h") or (c == 12 and crit == "probe z")
                    cellf(ws, r, c, v, fmt=f_, size=9, font=F, bold=hit, wrap=(c in (7, 8)))
                r += 1
        r += 1
    for cc, w in enumerate([9, 9, 14, 8, 26, 20, 26, 36, 6, 40] + [8] * (len(heads) - 10), start=1):
        ws.column_dimensions[get_column_letter(cc)].width = w
    ws.freeze_panes = ws.cell(row=5, column=6)


def failed_runs_sheet(wb, val_all, tdf_all):
    ws = wb.create_sheet("8 Failed runs")
    F = FULL_FONT
    cellf(ws, 1, 1, "Failed runs: collapsed (backbone probe below 60) and diverged (non-finite features)", bold=True, size=13, font=F)
    heads = ["Dataset", "Backbone", "Policy id", "Method", "Seed", "Run id", "Failure", "Val probe h", "Test probe h"]
    for j, h in enumerate(heads, start=1):
        cellf(ws, 3, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
    rows = {}
    for d, split in ((val_all, "val"), (tdf_all, "test")):
        if not len(d):
            continue
        bad = d[pd.to_numeric(d.probe_bb_bal_acc, errors="coerce") < COLLAPSE_BELOW]
        for x in bad.itertuples():
            rr = rows.setdefault(x.run_id, dict(ds=x.dataset, bb=x.backbone, pid=x.policy_id, o=x.objective, seed=x.seed,
                                                why="collapsed (probe below 60)", v=None, t=None))
            rr["v" if split == "val" else "t"] = num(x.probe_bb_bal_acc)
    for rid in sorted(read_diverged()):
        m = RUN_RE.match(rid)
        if not m or rid in rows:
            continue
        rows[rid] = dict(ds=m.group(3), bb={"resnet50": "ResNet-50", "resnet18": "ResNet-18"}.get(m.group(2), m.group(2)), pid=m.group(4),
                         o=m.group(1), seed=int(m.group(5) or 42), why="diverged (non-finite features)", v=None, t=None)
    r = 4
    for rid, x in sorted(rows.items()):
        row = [DNAME.get(x["ds"], x["ds"]), x["bb"], x["pid"], dict(OBJ).get(x["o"], x["o"]), int(x["seed"]), rid, x["why"], x["v"], x["t"]]
        for c, v in enumerate(row, start=1):
            cellf(ws, r, c, v, fmt="0.00" if c >= 8 else None, size=9, font=F)
        r += 1
    for cc, w in enumerate([10, 10, 24, 8, 6, 48, 30, 10, 10], start=1):
        ws.column_dimensions[get_column_letter(cc)].width = w


def methods_notes_sheet(wb, ok):
    ws = wb.create_sheet("9 Methods and notes")
    F = FULL_FONT
    floors = ok.groupby("dataset")["ms_floor"].mean() if "ms_floor" in ok.columns else pd.Series(dtype=float)
    fl = "; ".join(f"{DNAME[d]} {v:.4f}" for d, v in floors.items())
    tfl = "; ".join(f"{DNAME[d]} {pd.read_csv(f'study/test_ms_{d}.csv').ms_floor.iloc[0]:.4f}"
                    for d in ("octmnist", "pathmnist") if Path(f"study/test_ms_{d}.csv").exists()) or "not scored yet"
    sh = []
    for col, lab in (("purity_proj_shuf", "z"), ("purity_bb_shuf", "h")):
        if col in ok.columns:
            g = ok.groupby("dataset")[col].mean()
            sh.append(f"{lab}: " + ", ".join(f"{DNAME[d]} {v:.3f}" for d, v in g.items()))
    shtxt = "; ".join(sh)
    notes = [
        ("Data", "OCTMNIST and PathMNIST at 64 px. Validation = MedMNIST's validation split (class-unbalanced; for OCT it shares "
                 "patients with training). Test: OCT 1,000 images, 250 per class, new patients; PathMNIST 7,180 images from a different source."),
        ("Linear probes", "Backbone feature h (2048-d for ResNet-50) and projector output z (after the ReLU for NCL); LARS, 50 epochs, trained "
                          "on the full (unbalanced) training split without class weights; reported as balanced accuracy (%)."),
        ("k-NN-20", "Cosine k-NN on L2-normalised h with the training set as the bank, k = 20; balanced accuracy (%)."),
        ("SC %", "Class selectivity of projector units: for each unit, the share of its most frequent class among the images on which it is "
                 "non-zero; mean over units; on a class-balanced subset (smallest class size per class), 5 draws. Chance = 25% on OCT, "
                 "11.1% on PathMNIST. SimCLR's units are never zero, so its SC sits at chance by construction: SC compares NCL across policies."),
        ("SC above shuffle", "SC minus the same computation with shuffled labels, on the same scale (points). Removes what a unit's firing "
                             "frequency alone would give; the specialisation measure used for claims. Computed for every run, validation and test."),
        ("SC*", "(SC % − chance) / (1 − chance): a rescaling of SC %, carrying the same information within a dataset (in 'Runs' only)."),
        ("MS", "Activation-weighted mean DINOv2 (ViT-B/14) similarity of the images a unit responds to, on a fixed class-balanced subset of "
               "500 validation images per class; mean over units. Chance floor (a unit with no preference): " + fl + ". "
               "Also on test (eval_test_ms.py, fixed class-balanced test subset: 250 per class OCT, 339 PathMNIST), with its "
               "own floor: " + tfl + ". Compare raw MS within a split; use MS* to compare splits. "
               "MS* = (MS − floor) / (1 − floor), kept in the full table and 'Runs (val)'."),
        ("Purity", "For each unit, the share of the most frequent class among its 100 most strongly activating images; mean over units "
                   "that are above their minimum on more than 100 images; class-balanced subset, 5 draws. z = projector output, h = backbone. "
                   "Reported raw (0–1) in the main tables. Shuffled-label purity (chance), validation means: " + shtxt + ". "
                   "Purity* = (purity − shuffled) / (1 − shuffled), kept in the analysis sheets (4, 6, 7) and the full table. "
                   "The test files store purity* only, so test purity is reported as purity*. "
                   "Works for both methods, so it is the specialisation measure that compares NCL with SimCLR."),
        ("Effective dimension", "Participation ratio of the centred features' covariance eigenvalues: h = backbone, z = projector. Reported separately."),
        ("Dead projector units", "Projector units never non-zero on the validation set (NCL only in practice)."),
        ("Seeds and means", "Means and SDs (sample SD, n − 1) over successful runs; 'Attempted' = all trained runs; collapsed (backbone probe "
                            "below 60) and diverged (non-finite features) runs are listed in '8 Failed runs' and left out of means."),
        ("Noise rule", "Differences are marked beyond noise when larger than 2 SE, with SE from the seed SD pooled over the policies of the same "
                       "dataset, backbone, method and split. Not corrected for multiple comparisons; one-seed policies are a screen."),
        ("Selection", "'Best' sheets choose on validation only, separately per method and per probe; test values belong to the chosen run or policy."),
        ("Test plans", "Each test score was computed under a plan written before scoring (27 Oct, 2 Nov crop-only, and 2 Nov 'after analysis began' "
                       "to complete the table); see the last column of '0 Full table'."),
        ("Stem", "Old stem: random resized crop (area 0.08–1.0, 64 px) + brightness jitter 0.8 (p 0.8) + horizontal flip (p 0.5). Crop-only "
                 "base: the crop alone. Natural-image set: SimCLR's / NCL's default recipe. Pool augmentations are applied with p 0.5."),
        ("Projector size", "The projector-2048 runs change the architecture (output 2048 instead of 256); treated as a separate factor."),
    ]
    cellf(ws, 1, 1, "Methods and notes", bold=True, size=13, font=F)
    for i, (k, v) in enumerate(notes, start=3):
        cellf(ws, i, 1, k, bold=True, size=9, font=F, wrap=True)
        cellf(ws, i, 2, v, size=9, font=F, wrap=True)
    ws.column_dimensions["A"].width = 20; ws.column_dimensions["B"].width = 120


def sweeps_compact_sheet(wb, PS):
    """Sheet '5 Severity sweeps': one row per level, NCL and SimCLR side by side, the best level per family marked.
    Best = highest validation mean among levels with 2+ seeds, per method and probe; test is read off at that level.
    Also '5c Sweep charts': one line chart per family (validation and test probe h, both methods)."""
    from openpyxl.chart import LineChart, Reference
    from openpyxl.chart.series import SeriesLabel
    from openpyxl.chart.shapes import GraphicalProperties
    from openpyxl.drawing.line import LineProperties
    ws = wb.create_sheet("5 Severity sweeps")
    F = FULL_FONT
    ds, bb = "octmnist", "ResNet-50"
    cellf(ws, 1, 1, "Severity sweeps (OCT, ResNet-50): which strength works best, for NCL and SimCLR", bold=True, size=13, font=F)
    cellf(ws, 2, 1, "One row per level (means over valid seeds). Bold = best level on validation (highest mean among levels with 2+ seeds) "
                    "for that method and probe. Under each family, 'Best level' gives the validation choice in the validation columns and, "
                    "for comparison only, the best level on test in the test columns; 'Within noise of best' lists levels whose validation "
                    "mean is within 2 SE of the best (screening, not a test). Δ = NCL − SimCLR (green: NCL higher, pink: lower; "
                    "darker = beyond 2 SE). SDs and changes from the base level: '5a Sweeps detail'; charts: '5c Sweep charts'. "
                    "Brightness × flip has its own sheet (6).", size=9, italic=True, font=F, color=MUTED)
    PROBES = [("Val probe h", "val_h"), ("Test probe h", "test_h"), ("Val probe z", "val_z"), ("Test probe z", "test_z")]
    heads = ["Family", "Level", "Policy id", "Seeds (NCL / SimCLR)"]
    for lab, _k in PROBES:
        heads += [f"{lab}: NCL", f"{lab}: SimCLR", f"{lab}: Δ"]
    heads += ["Val SC %: NCL", "Val SC %: SimCLR", "Val MS: NCL", "Val MS: SimCLR", "Purity z: NCL", "Purity z: SimCLR"]
    for j, h in enumerate(heads, start=1):
        cellf(ws, 4, j, h, bold=True, fill=HEAD_FILL, wrap=True, size=9, font=F)
        ws.cell(row=4, column=j).border = Border(bottom=THIN)
    ws.row_dimensions[4].height = 42
    r = 5
    chart_specs = []
    for fam, base_name, levels in FAMILIES:
        if fam.startswith("Brightness ×"):
            continue
        title = f"{fam} — {base_name}"
        if fam.startswith("Projector"):
            title += " (architecture, a separate factor)"
        for c in range(1, len(heads) + 1):
            cellf(ws, r, c, fill=BLOCK_FILL, size=9, font=F)
        cellf(ws, r, 1, title, bold=True, fill=BLOCK_FILL, size=9, font=F)
        r += 1
        first = r
        stats = [(lab, pid, PS.get(ds, bb, pid, "ncl"), PS.get(ds, bb, pid, "simclr")) for lab, pid in levels]
        best, near, best_t = {}, {}, {}
        for lab_p, key in PROBES:
            for o, oi in (("ncl", 2), ("simclr", 3)):
                cand = [(st[oi][key][0], st[oi][key][2], st[0]) for st in stats if st[oi][key][0] is not None and st[oi][key][2] >= 2]
                if len(cand) < 2:
                    if key.startswith("val"):
                        best[(key, o)] = None
                    continue
                if key.startswith("val"):
                    m, n, lv = max(cand)
                    best[(key, o)] = lv
                    sd = _pooled(PS, ds, bb, o, key)
                    near[(key, o)] = [c_[2] for c_ in cand if c_[2] != lv and sd is not None
                                      and m - c_[0] <= 2 * sd * math.sqrt(1 / n + 1 / c_[1])]
                else:
                    best_t[(key, o)] = max(cand)[2]
        for lab, pid, stn, sts in stats:
            if stn["n_val"] == 0 and sts["n_val"] == 0 and stn["n_test"] == 0 and sts["n_test"] == 0:
                why = "diverged" if (stn["diverged"] or sts["diverged"]) else "no results yet"
                for j, v in enumerate([fam, lab, pid, why], start=1):
                    cellf(ws, r, j, v, size=9, font=F, italic=(j == 4), color=MUTED if j == 4 else None)
                r += 1
                continue
            fails = [f"{nm} {st[k]} {k}" for nm, st in (("NCL", stn), ("SimCLR", sts)) for k in ("collapsed", "diverged") if st[k]]
            seeds_txt = f"{stn['n_val']} / {sts['n_val']}" + (f" ({', '.join(fails)})" if fails else "")
            for cc, v in enumerate([fam, lab, pid, seeds_txt], start=1):
                cellf(ws, r, cc, v, size=9, font=F)
            c = 5
            for lab_p, key in PROBES:
                a, b = stn[key], sts[key]
                cellf(ws, r, c, a[0], fmt="0.00", size=9, font=F, bold=best.get((key, "ncl")) == lab)
                cellf(ws, r, c + 1, b[0], fmt="0.00", size=9, font=F, bold=best.get((key, "simclr")) == lab)
                if a[0] is not None and b[0] is not None:
                    d = a[0] - b[0]
                    cellf(ws, r, c + 2, d, fmt="+0.00;-0.00", italic=True, size=9, font=F, fill=_fill_z(d, PS.se(ds, bb, key, a[2], b[2])))
                c += 3
            for v, fm in ((stn["sc"][0], "0.0"), (sts["sc"][0], "0.0"), (stn["ms"][0], "0.000"), (sts["ms"][0], "0.000"),
                          (stn["pu_z_raw"][0], "0.000"), (sts["pu_z_raw"][0], "0.000")):
                cellf(ws, r, c, v, fmt=fm, size=9, font=F)
                c += 1
            r += 1
        last = r - 1
        cellf(ws, r, 2, "Best level", bold=True, size=9, font=F)
        cellf(ws, r + 1, 2, "Within noise of best", italic=True, size=9, font=F, color=MUTED)
        c = 5
        for lab_p, key in PROBES:
            for j, o in enumerate(("ncl", "simclr")):
                if key.startswith("val"):
                    if (key, o) in best and best[(key, o)] is None:
                        cellf(ws, r, c + j, "fewer than 2 levels with 2+ seeds", italic=True, size=9, font=F, color=MUTED)
                        continue
                    cellf(ws, r, c + j, best.get((key, o)), bold=True, size=9, font=F)
                    nn = near.get((key, o))
                    cellf(ws, r + 1, c + j, ", ".join(nn) if nn else ("none" if (key, o) in best else None), italic=True, size=9,
                          font=F, color=MUTED)
                else:
                    v = best_t.get((key, o))
                    cellf(ws, r, c + j, f"(test: {v})" if v else None, italic=True, size=9, font=F, color=MUTED)
            c += 3
        n_scored = sum(1 for st in stats if st[2]["val_h"][0] is not None or st[3]["val_h"][0] is not None)
        vals = [st[i][k][0] for st in stats for i in (2, 3) for k in ("val_h", "test_h") if st[i][k][0] is not None]
        if n_scored >= 2:
            chart_specs.append((title, first, last, math.floor(min(vals) / 5) * 5, math.ceil(max(vals) / 5) * 5))
        r += 3
    for cc, w in enumerate([22, 16, 22, 14] + [8.5] * 12 + [9] * 6, start=1):
        ws.column_dimensions[get_column_letter(cc)].width = w
    ws.freeze_panes = ws.cell(row=5, column=5)

    wc = wb.create_sheet("5c Sweep charts")
    cellf(wc, 1, 1, "Severity sweeps: probe h against strength (solid = validation, dashed = test). Data in '5 Severity sweeps'.",
          bold=True, size=11, font=F)
    colours = {"NCL": "1F5FAD", "SimCLR": "B5651D"}
    from openpyxl.drawing.text import CharacterProperties, ParagraphProperties
    for i, (title, first, last, ylo, yhi) in enumerate(chart_specs):
        ch = LineChart()
        ch.title = title
        tcp = CharacterProperties(sz=1100, b=True)
        ch.title.tx.rich.p[0].pPr = ParagraphProperties(defRPr=tcp)
        for run in ch.title.tx.rich.p[0].r:
            run.rPr = CharacterProperties(sz=1100, b=True)
        ch.height, ch.width = 8.0, 14.0
        ch.y_axis.scaling.min, ch.y_axis.scaling.max = ylo, yhi
        ch.y_axis.majorUnit = 5
        ch.y_axis.number_format = "0"
        ch.x_axis.delete = False
        ch.y_axis.delete = False
        ch.y_axis.title = "Balanced accuracy (%)"
        ch.x_axis.title = "Level"
        for col, name, meth, dash in ((5, "NCL, validation", "NCL", "solid"), (6, "SimCLR, validation", "SimCLR", "solid"),
                                      (8, "NCL, test", "NCL", "dash"), (9, "SimCLR, test", "SimCLR", "dash")):
            ch.add_data(Reference(ws, min_col=col, min_row=first, max_row=last), titles_from_data=False)
            sr = ch.series[-1]
            sr.tx = SeriesLabel(v=name)
            sr.graphicalProperties = GraphicalProperties(ln=LineProperties(solidFill=colours[meth], prstDash=dash, w=19050))
            sr.marker.symbol = "circle"
            sr.marker.size = 5
            sr.marker.graphicalProperties = GraphicalProperties(solidFill=colours[meth])
            sr.marker.graphicalProperties.line.solidFill = colours[meth]
            sr.smooth = False
        ch.set_categories(Reference(ws, min_col=2, min_row=first, max_row=last))
        ch.legend.position = "b"
        wc.add_chart(ch, f"{'A' if i % 2 == 0 else 'K'}{3 + (i // 2) * 18}")


def read_val(ds, backbone, folder):
    m, s = folder / "results_master.csv", folder / "results_spec.csv"
    if not m.exists():
        return None
    mm = pd.read_csv(m).drop_duplicates("run_id", keep="last")
    mm = mm.drop(columns=[c for c in ("sc", "sc_star", "ms") if c in mm.columns])
    if s.exists():
        ss = pd.read_csv(s).drop_duplicates("run_id", keep="last")
        keep = [c for c in ("run_id", "sc_star", "sc_star_shuf", "ms", "ms_floor", "ms_star", "purity_proj", "purity_proj_star",
                            "purity_bb", "purity_bb_star", "purity_proj_shuf", "purity_bb_shuf") if c in ss.columns]
        mm = mm.merge(ss[keep], on="run_id", how="left")
    mm["dataset"], mm["backbone"] = ds, backbone
    mm["objective"] = mm.run_id.str.split("-").str[0]
    return mm


def parse_rn18_log(path):
    rows, cur, which = [], None, None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^=== (\S+?)-offline-\S+-ep=99\.ckpt", line.strip())
        if m:
            cur = dict(run_id=m.group(1)); rows.append(cur); continue
        if "Linear Evaluation [backbone" in line:
            which = "bb"
        elif "Linear Evaluation [projector" in line:
            which = "proj"
        m = re.match(r"^Test\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%", line.strip())
        if m and cur is not None and which:
            cur[f"probe_{which}_acc"] = float(m.group(1)); cur[f"probe_{which}_bal_acc"] = float(m.group(2))
            cur[f"probe_{which}_auc"] = float(m.group(3))
    out = []
    for r in rows:
        rid = r["run_id"]
        cfg = "A6" if "-A6-" in rid else "A0"
        seed = int(rid.rsplit("-s", 1)[1]) if re.search(r"-s4\d$", rid) else 42
        out.append(dict(r, dataset="octmnist", backbone="ResNet-18", objective=rid.split("-")[0], seed=seed,
                        policy_id="stem" if cfg == "A0" else "speckle_contrast_crop020",
                        script="eval_linear_octmnist.py (probe seed 42)"))
    return out


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def cellf(ws, r, c, v=None, bold=False, size=10, italic=False, color=None, fill=None, fmt=None, wrap=False, align=None, font=None):
    cell = ws.cell(row=r, column=c)
    if v is not None:
        cell.value = v
    cell.font = Font(name=font or FONT, bold=bold, size=size, italic=italic, color=color)
    if fill:
        cell.fill = fill
    if fmt:
        cell.number_format = fmt
    if wrap or align:
        cell.alignment = Alignment(wrap_text=wrap, vertical="top" if wrap else "center", horizontal=align)
    return cell


def runs_header(ws, cols):
    ws.append(cols)
    for c in range(1, len(cols) + 1):
        cellf(ws, 1, c, bold=True, fill=HEAD_FILL, wrap=True)
        ws.cell(row=1, column=c).border = Border(bottom=THIN)


def main():
    used, missing, frames = [], [], []
    for ds, bb, folder in SOURCES:
        f = read_val(ds, bb, folder)
        if f is None:
            missing.append(f"{folder}/results_master.csv ({DNAME[ds]}, {bb})")
        else:
            frames.append(f); used.append(f"{folder}/results_master.csv + results_spec.csv ({DNAME[ds]}, {bb}): {len(f)} runs")
    val = pd.concat(frames, ignore_index=True)
    rn18 = val.backbone == "ResNet-18"
    val.loc[rn18, "policy_id"] = val.loc[rn18, "policy_id"].replace({"A0": "stem", "A6": "speckle_contrast_crop020"})

    test_rows = []
    for ds in ("octmnist", "pathmnist"):
        p = Path(f"study/test_results_{ds}.csv")
        if p.exists():
            t = pd.read_csv(p)
            t["backbone"] = t["backbone"].map({"resnet50": "ResNet-50", "resnet18": "ResNet-18"}).fillna(t["backbone"])
            t["script"] = "eval_test.py"
            pm = Path(f"study/test_ms_{ds}.csv")              # test MS (eval_test_ms.py), when scored
            if pm.exists():
                ms = pd.read_csv(pm)[["run_id", "ms", "ms_star", "ms_floor"]].rename(
                    columns={"ms": "test_ms", "ms_star": "test_ms_star", "ms_floor": "test_ms_floor"})
                t = t.merge(ms, on="run_id", how="left")
                used.append(f"{pm}: {len(ms)} runs")
            test_rows += t.to_dict("records"); used.append(f"{p}: {len(t)} runs")
        else:
            missing.append(f"{p} (eval_test.py, {DNAME[ds]})")
    lp = Path("study/rn18/test_probe.log")
    if lp.exists():
        have = {r["run_id"] for r in test_rows}
        rr = [r for r in parse_rn18_log(lp) if r["run_id"] not in have]
        test_rows += rr; used.append(f"{lp}: {len(rr)} ResNet-18 runs")
    else:
        missing.append(f"{lp} (ResNet-18 test probes)")

    wb = Workbook()
    readme = wb.active; readme.title = "Read me"
    ws_s = wb.create_sheet("1 Projector only")
    ws_p = wb.create_sheet("2 Backbone acc + projector mono")
    ws_a = wb.create_sheet("3 Augmentations")
    ws_m = wb.create_sheet("More policies")
    ws_v = wb.create_sheet("Runs (val)")
    ws_t = wb.create_sheet("Runs (test)")

    # ---------------- Runs (val): one row per run
    vcols = ["Dataset", "Backbone", "Policy id", "Objective", "Seed", "Run id", "Classes", "Collapsed",
             "Probe backbone", "Probe projector", "k-NN-5", "k-NN-20", "k-NN-100", "SC*", "SC %",
             "MS", "MS floor", "MS*", "Purity z", "Purity z*", "Purity h", "Purity h*", "Dead units",
             "Eff. dim h (backbone)", "Eff. dim z (projector)", "SC above shuffle"]
    runs_header(ws_v, vcols)
    val["_k"] = val.apply(lambda r: (r.dataset, r.backbone, policy_info(r.policy_id, r.dataset)[2], r.objective, r.seed), axis=1)
    val = val.sort_values("_k")
    for i, r in enumerate(val.itertuples(), start=2):
        g = lambda c: num(getattr(r, c, None))
        probe, scs, C = g("probe_bb_bal_acc"), g("sc_star"), CLASSES[r.dataset]
        ws_v.append([DNAME[r.dataset], r.backbone, r.policy_id, r.objective,
                     int(r.seed), r.run_id, C, "yes" if probe is not None and probe < COLLAPSE_BELOW else "no",
                     probe, g("probe_proj_bal_acc"), g("knn5_bal_acc"), g("knn20_bal_acc"), g("knn100_bal_acc"),
                     scs, None if scs is None else 100 / C + (1 - 1 / C) * scs, g("ms"), g("ms_floor"), g("ms_star"),
                     g("purity_proj"), g("purity_proj_star"), g("purity_bb"), g("purity_bb_star"), g("dead_proj"), g("effdim_bb"),
                     g("effdim_proj"), None if scs is None or g("sc_star_shuf") is None else (1 - 1 / C) * (scs - g("sc_star_shuf"))])
    nv = len(val) + 1
    fm = {9: "0.00", 10: "0.00", 11: "0.0", 12: "0.0", 13: "0.0", 14: "0.0", 15: "0.0", 16: "0.000", 17: "0.000",
          18: "0.000", 19: "0.000", 20: "0.000", 21: "0.000", 22: "0.000", 23: "0", 24: "0.0", 25: "0.0", 26: "0.0"}
    for rr in range(2, nv + 1):
        for c in range(1, len(vcols) + 1):
            cellf(ws_v, rr, c, fmt=fm.get(c))
    for c, w in zip(range(1, len(vcols) + 1), [10, 11, 30, 9, 6, 52, 8, 10] + [11] * 18):
        ws_v.column_dimensions[get_column_letter(c)].width = w
    ws_v.freeze_panes = "G2"

    # ---------------- Runs (test): one row per run
    tcols = ["Dataset", "Backbone", "Policy id", "Objective", "Seed", "Run id", "Script",
             "Probe backbone", "Probe projector", "AUC backbone", "k-NN-20", "SC*", "Purity z*", "Recall per class, backbone"]
    runs_header(ws_t, tcols)
    test_rows.sort(key=lambda r: (r["dataset"], r["backbone"], policy_info(r["policy_id"], r["dataset"])[2], r["objective"], int(r["seed"])))
    for i, r in enumerate(test_rows, start=2):
        ws_t.append([DNAME[r["dataset"]], r["backbone"], r["policy_id"],
                     r["objective"], int(r["seed"]), r["run_id"], r.get("script", ""), num(r.get("probe_bb_bal_acc")),
                     num(r.get("probe_proj_bal_acc")), num(r.get("probe_bb_auc")), num(r.get("knn20_bal_acc")),
                     num(r.get("sc_star")), num(r.get("purity_proj_star")), r.get("recall_bb", "") or ""])
    nt = max(len(test_rows) + 1, 2)
    for rr in range(2, nt + 1):
        for c in range(1, len(tcols) + 1):
            cellf(ws_t, rr, c, fmt={8: "0.00", 9: "0.00", 10: "0.00", 11: "0.0", 12: "0.0", 13: "0.000"}.get(c))
    for c, w in zip(range(1, len(tcols) + 1), [10, 11, 30, 9, 6, 52, 30, 12, 12, 11, 11, 10, 11, 30]):
        ws_t.column_dimensions[get_column_letter(c)].width = w
    ws_t.freeze_panes = "G2"
    if not test_rows:
        cellf(ws_t, 2, 1, "No test results yet: run eval_test.py, copy the results to the Mac and run this script again.", italic=True)

    # data for the summary sheets: values computed here from the csv files (no spreadsheet formulas)
    val["sc_pct"] = val.apply(lambda r: 100 / CLASSES[r.dataset] + (1 - 1 / CLASSES[r.dataset]) * r.sc_star
                              if pd.notna(r.sc_star) else float("nan"), axis=1)
    val["sc_above"] = val.apply(lambda r: (1 - 1 / CLASSES[r.dataset]) * (r.sc_star - r.sc_star_shuf)
                                if pd.notna(r.sc_star) and pd.notna(r.get("sc_star_shuf")) else float("nan"), axis=1)
    ok = val[~(val.probe_bb_bal_acc < COLLAPSE_BELOW)]
    tdf_all = pd.DataFrame(test_rows) if test_rows else pd.DataFrame(columns=["dataset", "backbone", "policy_id", "objective"])
    # collapsed runs (backbone probe below COLLAPSE_BELOW on validation or on test) are left out of the test means too
    coll_ids = set(val.loc[val.probe_bb_bal_acc < COLLAPSE_BELOW, "run_id"])
    if len(tdf_all) and "probe_bb_bal_acc" in tdf_all.columns:
        coll_ids |= set(tdf_all.loc[pd.to_numeric(tdf_all.probe_bb_bal_acc, errors="coerce") < COLLAPSE_BELOW, "run_id"])
        tdf = tdf_all[~tdf_all.run_id.isin(coll_ids)]
    else:
        tdf = tdf_all
    if len(tdf) and "sc_star" in tdf.columns:
        tdf = tdf.copy()
        tdf["sc_pct"] = [100 / CLASSES[d] + (1 - 1 / CLASSES[d]) * float(v) if pd.notna(v) else float("nan")
                         for d, v in zip(tdf.dataset, pd.to_numeric(tdf.sc_star, errors="coerce"))]
        if "sc_star_shuf" in tdf.columns:
            tdf["sc_above"] = [(1 - 1 / CLASSES[d]) * (float(a) - float(b)) if pd.notna(a) and pd.notna(b) else float("nan")
                               for d, a, b in zip(tdf.dataset, pd.to_numeric(tdf.sc_star, errors="coerce"),
                                                  pd.to_numeric(tdf.sc_star_shuf, errors="coerce"))]

    def mean_of(df, col):
        x = pd.to_numeric(df[col], errors="coerce").dropna() if col in df else pd.Series(dtype=float)
        return float(x.mean()) if len(x) else None

    # policies, one row each (dataset, backbone, policy)
    keys = val.groupby(["dataset", "backbone", "policy_id"]).size().reset_index()
    keys["_k"] = keys.apply(lambda r: (r.dataset, r.backbone != "ResNet-50", policy_info(r.policy_id, r.dataset)[2]), axis=1)
    keys = keys.sort_values("_k")
    keys["main"] = keys.apply(lambda r: policy_info(r.policy_id, r.dataset)[1], axis=1)

    def side_sheet(ws, title, note, metrics, rows, widths=None):
        """One row per policy; each metric gets an NCL and a SimCLR column (mean over seeds)."""
        cellf(ws, 1, 2, title, bold=True, size=13)
        cellf(ws, 2, 2, note, size=9, italic=True, color=MUTED)
        ncol = 3 + 2 * len(metrics)
        cellf(ws, 4, 2, "Policy", bold=True, fill=HEAD_FILL); cellf(ws, 5, 2, None, fill=HEAD_FILL)
        cellf(ws, 4, 3, "Seeds", bold=True, fill=HEAD_FILL, align="center")
        cellf(ws, 5, 3, "NCL / SimCLR", size=9, color=MUTED, fill=HEAD_FILL, align="center")
        ws.merge_cells(start_row=4, start_column=2, end_row=5, end_column=2)
        for j, (label, *_rest) in enumerate(metrics):
            c0 = 4 + 2 * j
            cellf(ws, 4, c0, label, bold=True, fill=HEAD_FILL, align="center"); cellf(ws, 4, c0 + 1, None, fill=HEAD_FILL)
            ws.merge_cells(start_row=4, start_column=c0, end_row=4, end_column=c0 + 1)
            for k, (_o, oname) in enumerate(OBJ):
                cellf(ws, 5, c0 + k, oname, bold=True, size=9, fill=HEAD_FILL, align="center")
        for c in range(2, ncol + 1):
            ws.cell(row=5, column=c).border = Border(bottom=THIN)
        r, prev = 6, None
        for k in rows.itertuples():
            block = (k.dataset, k.backbone)
            if block != prev:
                if prev is not None:
                    r += 1
                for c in range(2, ncol + 1):
                    cellf(ws, r, c, fill=BLOCK_FILL)
                cellf(ws, r, 2, f"{DNAME[k.dataset]}, {k.backbone}", bold=True, fill=BLOCK_FILL)
                r += 1; prev = block
            name = policy_info(k.policy_id, k.dataset)[0]
            if k.backbone == "ResNet-18":
                name = {"stem": "A0 (stem)"}.get(k.policy_id, name)
            stem = k.policy_id in ("stem", "A0")
            fill = STEM_FILL if stem else None
            ws.cell(row=r, column=1, value=f"{DNAME[k.dataset]}|{k.backbone}|{k.policy_id}")
            cellf(ws, r, 2, name, bold=stem, fill=fill)
            sel = ok[(ok.dataset == k.dataset) & (ok.backbone == k.backbone) & (ok.policy_id == k.policy_id)]
            tsel = tdf[(tdf.dataset == k.dataset) & (tdf.backbone == k.backbone) & (tdf.policy_id == k.policy_id)] if len(tdf) else tdf
            # runs counted (not collapsed); if some of them have no SC/MS/purity yet, say how many do
            n = {o: int((sel.objective == o).sum()) for o, _ in OBJ}
            sm = {o: int(((sel.objective == o) & sel.sc_star.notna()).sum()) for o, _ in OBJ}
            seeds = f"{n['ncl']} / {n['simclr']}" + ("" if sm == n else f" (mono {sm['ncl']} / {sm['simclr']})")
            cellf(ws, r, 3, seeds, fill=fill, align="center")
            for j, (_label, src, col, fmt) in enumerate(metrics):
                for kk, (o, _on) in enumerate(OBJ):
                    d = sel[sel.objective == o] if src == "val" else (tsel[tsel.objective == o] if len(tsel) else tsel)
                    cellf(ws, r, 4 + 2 * j + kk, mean_of(d, col), fmt=fmt, fill=fill, bold=stem)
            r += 1
        ws.column_dimensions["A"].hidden = True
        ws.column_dimensions["B"].width = 34
        ws.column_dimensions["C"].width = 22
        for c in range(4, ncol + 1):
            ws.column_dimensions[get_column_letter(c)].width = (widths or 10)
        ws.merge_cells(start_row=2, start_column=2, end_row=2, end_column=ncol)
        ws.row_dimensions[2].height = 40
        ws.cell(row=2, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "C6"
        return r - 1

    main_rows, more_rows = keys[keys.main], keys[~keys.main]
    spec = [("SC %", "val", "sc_pct", "0.0"), ("MS", "val", "ms", "0.000"), ("Class purity", "val", "purity_proj_star", "0.00")]
    spec_note = ("SC, MS and class purity are measured on the projector output, validation. SC chance: 25% OCT, 11.1% PathMNIST. "
                 "MS floor (no preference): 0.604 OCT. Class purity: majority-class share of each projector unit's top 100 images, "
                 "corrected with shuffled labels (0 = chance, 1 = one class). Mean over seeds; shaded row = stem, the reference.")
    side_sheet(ws_s, "1 Projector only: accuracy and monosemanticity on the projector output",
               "Linear probe = balanced accuracy (%) of a linear classifier on the frozen projector output z (NCL after its ReLU), "
               "as in BYI. " + spec_note,
               [("Probe z, val", "val", "probe_proj_bal_acc", "0.0"), ("Probe z, test", "test", "probe_proj_bal_acc", "0.0")] + spec, main_rows)
    side_sheet(ws_p, "2 Backbone accuracy, projector monosemanticity",
               "Linear probe and k-NN-20 = balanced accuracy (%) on the frozen backbone output h, as in NCL. " + spec_note,
               [("Probe h, val", "val", "probe_bb_bal_acc", "0.0"), ("Probe h, test", "test", "probe_bb_bal_acc", "0.0"),
                ("k-NN-20 h, val", "val", "knn20_bal_acc", "0.0")] + spec,
               main_rows)
    side_sheet(ws_m, "More policies (OCT screen)",
               "Pairs, five-operation stacks and strength variants. Mean over seeds, validation; most have one seed. "
               "Probe z = projector output, probe h = backbone output.",
               [("Probe z", "val", "probe_proj_bal_acc", "0.0"), ("Probe h", "val", "probe_bb_bal_acc", "0.0")] + spec, more_rows)

    # ---------------- 3 Augmentations: one table
    cellf(ws_a, 1, 2, "3 Augmentations", bold=True, size=13)
    cellf(ws_a, 2, 2, "Every policy = the stem plus its operations, each applied with probability 0.5 per view at the strength below. "
                      "Strengths come from the label test (milder ladders for blur, motion, pixelation and JPEG on OCT); contrast uses "
                      "0.8, the A6 value; PathMNIST reuses the OCT values for contrast, pixelation and JPEG.",
          size=9, italic=True, color=MUTED)
    hdr = ["Augmentation", "What it does", "OCT", "PathMNIST"]
    for j, h in enumerate(hdr, start=2):
        cellf(ws_a, 4, j, h, bold=True, fill=HEAD_FILL); ws_a.cell(row=4, column=j).border = Border(bottom=THIN)
    ops = [
        ("Stem (in every policy)", "random resized crop to 64 px, minimum area 0.08; brightness jitter 0.2 to 1.8, p 0.8; horizontal flip p 0.5", "on", "on"),
        ("Speckle", "multiplicative noise: x + x·ε, ε ~ N(0, std²)", "std 0.1", "—"),
        ("Contrast", "pulls pixels towards the image mean: μ + f·(x − μ)", "factor 0.8", "factor 0.8"),
        ("Blur", "Gaussian blur, σ up to a maximum", "σ max 1.2", "—"),
        ("Motion", "motion-blur kernel, ±15°", "radius 5, σ 2", "—"),
        ("Pixelation", "down-sample, then up-sample", "factor 0.8", "factor 0.8"),
        ("JPEG", "save and reload at a quality", "quality 70", "quality 70"),
        ("Rotation + flip", "random multiple of 90°; vertical flip p 0.5", "—", "on"),
        ("Stain", "jitters H&E optical density: c·a + b, a ~ U[1−σ, 1+σ], b ~ U[−σ, σ]", "—", "σ 0.2"),
        ("Defocus", "disk-kernel blur", "—", "radius 3"),
        ("All six", "every operation in the dataset's pool", "on", "on"),
        ("Natural-image set", "SimCLR's ImageNet recipe (Chen et al. 2020): colour jitter 0.8/0.8/0.8/0.2 p 0.8, grayscale 0.2, Gaussian blur 0.5; replaces the stem's brightness jitter, no pool operations added", "on", "on"),
        ("A6", "speckle + contrast with crop minimum area 0.2", "on", "—"),
    ]
    for i, o in enumerate(ops, start=5):
        for j, v in enumerate(o, start=2):
            cellf(ws_a, i, j, v, wrap=True, bold=(j == 2))
    for col, w in zip("ABCDE", [2, 22, 62, 14, 14]):
        ws_a.column_dimensions[col].width = w

    # ---------------- Read me
    cellf(readme, 1, 2, "Results workbook", bold=True, size=14)
    lines = [
        ("0 Full table", "Everything in one table: one row per dataset, backbone, policy and method; the augmentations applied (read from the "
                         "training files, study/derived/policy_augs.csv); every seed's backbone probe, SC % and projector purity with their "
                         "mean and SD; the other validation means; the difference in probe from the reference policy (stem, or crop only "
                         "for the crop-only policies); and the test-split means. Collapsed (probe below 60) and diverged runs are left out. "
                         "The last column names the test plan each row was scored under: the 27 Oct and 2 Nov crop-only plans were "
                         "written before the analysis; rows marked 'added 2 Nov, after analysis began' were scored once the analysis "
                         "had started, to complete the table. MS is given raw for every seed, with its mean, SD and "
                         "the dataset's chance floor (the MS of a unit with no preference); MS* = (MS - floor) / (1 - floor). "
                         "The crop-only block also shows the stem as 'Crop only + brightness + flip (= stem)': "
                         "the same runs as the old-stem screen's Stem row, compared here with crop only, so the block holds the full "
                         "brightness × flip 2×2. "
                         "Also written to study/summary/full_table.csv."),
        ("0b NCL ahead of SimCLR", "Every policy where NCL's mean is above SimCLR's on at least one linear probe (backbone or "
                                   "projector, validation or test): both methods' means and SDs, the difference and its z, NCL's SC, "
                                   "and both methods' purity. Also written to study/summary/ncl_ahead.csv."),
        ("1, 2 Best validation", "For each method separately (NCL, then SimCLR): the best single run and the best policy (mean over 3+ "
                                 "seeds), chosen on validation by probe h and by probe z, with the same run's or policy's test results, "
                                 "the stem / base, the augmentations applied, and the policy's attempted, valid and failed runs."),
        ("3 Matched method comparison", "NCL and SimCLR under the same policy: rows NCL, SimCLR and NCL \u2212 SimCLR, with means and SDs "
                                        "in separate columns (probes h and z, k-NN-20, SC %, SC above shuffle, raw MS, raw purity, "
                                        "effective dimension h and z, test values)."),
        ("4 Augmentation effects", "Each policy's change from its reference (stem, or crop only), per method, and NCL's change minus "
                                   "SimCLR's: whether an augmentation acts differently under non-negativity. Purity as purity*."),
        ("5 Severity sweeps", "One augmentation varied at a time, one row per level, NCL and SimCLR side by side (probes h and z on "
                              "validation and test, NCL's SC above shuffle and MS, purity z), with the best level per family marked "
                              "(validation mean, 2+ seeds) and the levels within noise of it. 5a: full detail with SDs and changes "
                              "from the base level. 5b: the best seed per level. 5c: a chart per family."),
        ("6 Brightness \u00d7 flip", "The replicated 2\u00d72 on the crop-only base: cell means, the two main effects and the interaction "
                                     "with its SE and z, per method and metric."),
        ("7 Specialisation & utilisation", "SC %, SC above shuffle, raw MS and MS*, raw purity and purity* (z and h), effective dimension "
                                           "h and z, dead projector units, per policy and method."),
        ("8 Runs (val), 8 Runs (test), 8 Failed runs", "One row per run, copied from the csv files (every sheet's numbers are computed "
                                                       "from these rows); the failed runs with the reason."),
        ("9 Methods and notes", "Metric definitions, chance levels and floors per dataset, selection rules, the noise rule, exclusions."),
        ("x sheets", "The earlier side sheets (projector only, backbone + projector, augmentation descriptions, probe per seed, "
                     "more policies), kept for reference."),
        ("Collapsed runs", f"Runs with a backbone probe below {COLLAPSE_BELOW:.0f} are left out of the validation and test means: "
                           + (", ".join(sorted(coll_ids)) if coll_ids else "none") + "."),
        ("Fills, 0 Full table", "On each policy's NCL row: green where NCL's mean is above SimCLR's, rose where below, darker where "
                                "the difference is beyond 2 SE (seed SD pooled over policies). Policy cell gold where the story holds "
                                "on validation or test: NCL at least level with SimCLR on both the backbone and the projector probe, "
                                "with higher projector purity; the note on the cell gives the numbers. SC cells: NCL's level, blue, "
                                "darker = more specialised. A key sits to the right of the table."),
        ("Generated", datetime.now().strftime("%d %b %Y %H:%M") + " by study_tools/make_results_workbook.py"),
        ("Not found yet", "; ".join(missing) if missing else "none"),
    ]
    cellf(readme, 3, 2, "Key: fills in 0 Full table", bold=True)
    cellf(readme, 3, 3, "On each policy's NCL row (SimCLR rows are left plain). The same key sits to the right of the table, in columns AT and AU.", wrap=True)
    i = 4
    for fc, text in FILL_KEY[1:]:
        if fc:
            cellf(readme, i, 2, None, fill=PatternFill("solid", start_color=fc))
        cellf(readme, i, 3, text, wrap=True)
        i += 1
    for i, (k, v) in enumerate(lines, start=i + 1):
        cellf(readme, i, 2, k, bold=True, wrap=True)
        cellf(readme, i, 3, v, wrap=True)
    readme.column_dimensions["A"].width = 2; readme.column_dimensions["B"].width = 22; readme.column_dimensions["C"].width = 100

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # ---------------- 4 Probe per seed: every seed's linear-probe accuracy, then the mean and SD over seeds
    ws_q = wb.create_sheet("4 Probe per seed", 4)
    qmetrics = [("Backbone probe, val", "val", "probe_bb_bal_acc"), ("Projector probe, val", "val", "probe_proj_bal_acc"),
                ("Backbone probe, test", "test", "probe_bb_bal_acc"), ("Projector probe, test", "test", "probe_proj_bal_acc")]
    qseeds = sorted({int(x) for x in pd.to_numeric(val["seed"], errors="coerce").dropna()} |
                    ({int(x) for x in pd.to_numeric(tdf_all["seed"], errors="coerce").dropna()} if len(tdf_all) and "seed" in tdf_all else set()))
    cellf(ws_q, 1, 1, "Linear-probe balanced accuracy (%) for every seed, then the mean and SD over seeds", bold=True, size=13)
    cellf(ws_q, 2, 1, f"Each seed's value as scored. Mean and SD leave out collapsed runs (backbone probe below "
                      f"{COLLAPSE_BELOW:.0f}), which are shown in grey italics. SD is the sample SD (n - 1). "
                      f"Empty cell: not trained or not scored yet.", size=9, italic=True, color=MUTED)
    qheads = ["Dataset", "Backbone", "Policy", "Method"]
    for j, h in enumerate(qheads, start=1):
        cellf(ws_q, 4, j, h, bold=True, fill=HEAD_FILL); cellf(ws_q, 5, j, None, fill=HEAD_FILL)
    c = len(qheads) + 1
    for label, _src, _col in qmetrics:
        cellf(ws_q, 4, c, label, bold=True, fill=HEAD_FILL, align="center")
        for kq in range(1, 5):
            cellf(ws_q, 4, c + kq, None, fill=HEAD_FILL)
        ws_q.merge_cells(start_row=4, start_column=c, end_row=4, end_column=c + 4)
        for kq, sub in enumerate([f"seed {sd}" for sd in qseeds] + ["Mean", "SD"]):
            cellf(ws_q, 5, c + kq, sub, bold=True, size=9, fill=HEAD_FILL, align="center")
        c += 5
    for cq in range(1, c):
        ws_q.cell(row=5, column=cq).border = Border(bottom=THIN)
    rq = 6
    for k in keys.itertuples():
        for o, oname in OBJ:
            vsel = val[(val.dataset == k.dataset) & (val.backbone == k.backbone) & (val.policy_id == k.policy_id) & (val.objective == o)]
            tsel = (tdf_all[(tdf_all.dataset == k.dataset) & (tdf_all.backbone == k.backbone) & (tdf_all.policy_id == k.policy_id)
                        & (tdf_all.objective == o)] if len(tdf_all) else tdf_all)
            if vsel.empty and len(tsel) == 0:
                continue
            collapsed = {int(sd) for sd, pv in zip(vsel.seed, vsel.probe_bb_bal_acc) if pd.notna(pv) and pv < COLLAPSE_BELOW}
            stem = k.policy_id in ("stem", "A0")
            cellf(ws_q, rq, 1, DNAME[k.dataset]); cellf(ws_q, rq, 2, k.backbone)
            cellf(ws_q, rq, 3, policy_info(k.policy_id, k.dataset)[0], bold=stem); cellf(ws_q, rq, 4, oname)
            c = len(qheads) + 1
            for _label, src, col in qmetrics:
                d = vsel if src == "val" else tsel
                vals = []
                for sd in qseeds:
                    v = None
                    if len(d) and col in d.columns and "seed" in d.columns:
                        x = pd.to_numeric(d[pd.to_numeric(d["seed"], errors="coerce") == sd][col], errors="coerce").dropna()
                        v = float(x.iloc[-1]) if len(x) else None
                    gone = sd in collapsed
                    cellf(ws_q, rq, c, v, fmt="0.00", color=MUTED if gone else None, italic=gone)
                    if v is not None and not gone:
                        vals.append(v)
                    c += 1
                mq = sum(vals) / len(vals) if vals else None
                sdq = math.sqrt(sum((x - mq) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else None
                cellf(ws_q, rq, c, mq, fmt="0.00", bold=True); cellf(ws_q, rq, c + 1, sdq, fmt="0.00")
                c += 2
            rq += 1
    ws_q.column_dimensions["A"].width = 11; ws_q.column_dimensions["B"].width = 11
    ws_q.column_dimensions["C"].width = 34; ws_q.column_dimensions["D"].width = 9
    for cq in range(5, c):
        ws_q.column_dimensions[get_column_letter(cq)].width = 9
    ws_q.freeze_panes = "E6"

    # the stem is the crop-only design's fourth cell (crop 0.08 + brightness 0.8 + flip): shown again in the crop-only block,
    # under its own id, with crop only as its reference (the runs are the stem's; nothing is re-scored)
    def with_stem_cell(df):
        if not len(df):
            return df
        st = df[(df.dataset == "octmnist") & (df.backbone == "ResNet-50") & (df.policy_id == "stem")].copy()
        st["policy_id"] = "croponly_stem"
        return pd.concat([df, st], ignore_index=True)
    val2, tdf2 = with_stem_cell(val), with_stem_cell(tdf)
    ok2 = val2[~(val2.probe_bb_bal_acc < COLLAPSE_BELOW)]
    keys2 = val2.groupby(["dataset", "backbone", "policy_id"]).size().reset_index()
    keys2["_k"] = keys2.apply(lambda r: (r.dataset, r.backbone != "ResNet-50", policy_info(r.policy_id, r.dataset)[2]), axis=1)
    keys2 = keys2.sort_values("_k")
    keys2["main"] = keys2.apply(lambda r: policy_info(r.policy_id, r.dataset)[1], axis=1)
    full_table(wb, val2, tdf2, keys2, qseeds)
    n_ahead = ncl_ahead_sheet(wb, ok2, tdf2, keys2)
    PS = PolicyStats(val, ok, tdf)
    augs_all = read_policy_augs()
    best_method_sheet(wb, PS, "ncl", "1 Best validation — NCL", augs_all)
    best_method_sheet(wb, PS, "simclr", "2 Best validation — SimCLR", augs_all)
    all_policy_means_sheet(wb, PS, keys, augs_all)
    augmentation_effects_sheet(wb, PS, keys2, augs_all)
    sweeps_compact_sheet(wb, PS)
    sweeps_sheet(wb, PS, augs_all)
    sweep_best_sheet(wb, PS)
    brightness_flip_sheet(wb, PS)
    all_policy_means_sheet(wb, PS, keys, augs_all, name="7 Specialisation & utilisation", cols=UTIL_COLS,
                           title="Specialisation and utilisation: SC, MS, purity, effective dimension and dead units, NCL and SimCLR")
    failed_runs_sheet(wb, val, tdf_all)
    methods_notes_sheet(wb, ok)
    # order: Read me, 0, 0b, 1-7, 8 (runs), 9, then the older side sheets
    old = {"1 Projector only": "x Projector only", "2 Backbone acc + projector mono": "x Backbone acc + projector mono",
           "3 Augmentations": "x Augmentation descriptions", "4 Probe per seed": "x Probe per seed", "More policies": "x More policies",
           "Runs (val)": "8 Runs (val)", "Runs (test)": "8 Runs (test)"}
    for a_, b_ in old.items():
        if a_ in wb.sheetnames:
            wb[a_].title = b_
    order = ["Read me", "0 Full table", "0b NCL ahead of SimCLR", "1 Best validation — NCL", "2 Best validation — SimCLR",
             "3 Matched method comparison", "4 Augmentation effects", "5 Severity sweeps", "5a Sweeps detail", "5b Sweep best seeds",
             "5c Sweep charts", "6 Brightness × flip",
             "7 Specialisation & utilisation", "8 Runs (val)", "8 Runs (test)", "8 Failed runs", "9 Methods and notes"]
    rest = [n for n in wb.sheetnames if n not in order]
    wb._sheets = [wb[n] for n in order if n in wb.sheetnames] + [wb[n] for n in rest]
    wb.save(OUT)
    print(f"wrote {OUT}: {nv - 1} validation runs, {len(test_rows)} test runs, "
          f"{int(keys.main.sum())} main policies, {int((~keys.main).sum())} more")
    for m in missing:
        print("  not found:", m)


if __name__ == "__main__":
    main()
