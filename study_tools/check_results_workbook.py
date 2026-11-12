"""Independent check of results_workbook.xlsx against the raw csv files (does not import the workbook script)."""
import math, re
import openpyxl, pandas as pd, numpy as np

WB = "study/summary/results_workbook.xlsx"
C = {"OCT": 4, "PathMNIST": 9}
SRC = [("OCT", "ResNet-50", "study"), ("OCT", "ResNet-18", "study/rn18"), ("PathMNIST", "ResNet-50", "study/path")]

# ---- raw data
val = []
for ds, bb, d in SRC:
    m = pd.read_csv(f"{d}/results_master.csv"); s = pd.read_csv(f"{d}/results_spec.csv")
    for name, f in (("master", m), ("spec", s)):
        dup = f[f.run_id.duplicated(keep=False)]
        if len(dup):
            print(f"[dup] {d}/{name}: {dup.run_id.nunique()} run ids appear more than once")
            for rid, g in dup.groupby("run_id"):
                cols = [c for c in ("probe_bb_bal_acc", "sc_star", "ms", "purity_proj_star") if c in g]
                if len(g[cols].drop_duplicates()) > 1:
                    print("      differing values:", rid, g[cols + [c for c in ("eval_date",) if c in g]].values.tolist())
    m = m.drop_duplicates("run_id", keep="last"); s = s.drop_duplicates("run_id", keep="last")
    m = m[["run_id", "seed", "policy_id", "probe_bb_bal_acc", "probe_proj_bal_acc", "knn20_bal_acc"]]
    s = s[["run_id", "sc_star", "ms", "purity_proj_star"]]
    f = m.merge(s, on="run_id", how="left")
    f["dataset"], f["backbone"] = ds, bb
    f["objective"] = f.run_id.str.split("-").str[0]
    f["policy_id"] = f.policy_id.replace({"A0": "stem", "A6": "speckle_contrast_crop020"})
    f["sc_pct"] = 100 / C[ds] + (1 - 1 / C[ds]) * f.sc_star
    val.append(f)
val = pd.concat(val, ignore_index=True)
val["collapsed"] = val.probe_bb_bal_acc < 60
ok = val[~val.collapsed]

t = pd.read_csv("study/test_results_octmnist.csv")
t["dataset"] = "OCT"; t["backbone"] = t.backbone.map({"resnet50": "ResNet-50", "resnet18": "ResNet-18"})

VAL_COL = {"SC %": "sc_pct", "MS": "ms", "Class purity": "purity_proj_star", "Probe z, val": "probe_proj_bal_acc",
           "Probe z": "probe_proj_bal_acc", "Probe h, val": "probe_bb_bal_acc", "Probe h": "probe_bb_bal_acc",
           "k-NN-20 h, val": "knn20_bal_acc"}
TEST_COL = {"Probe z, test": "probe_proj_bal_acc", "Probe h, test": "probe_bb_bal_acc"}


def expected(ds, bb, pid, obj, label):
    if label in TEST_COL:
        x = t[(t.dataset == ds) & (t.backbone == bb) & (t.policy_id == pid) & (t.objective == obj)][TEST_COL[label]]
    else:
        x = ok[(ok.dataset == ds) & (ok.backbone == bb) & (ok.policy_id == pid) & (ok.objective == obj)][VAL_COL[label]]
    x = x.dropna()
    return (float(x.mean()), len(x)) if len(x) else (None, 0)


wb = openpyxl.load_workbook(WB, data_only=True)
total = bad = 0
for sheet in ("1 Projector only", "2 Backbone acc + projector mono", "More policies"):
    ws = wb[sheet]
    top = [c.value for c in ws[4]]; sub = [c.value for c in ws[5]]
    labels, cur = [], None
    for j in range(len(top)):
        cur = top[j] if top[j] is not None else cur
        labels.append(cur)
    for row in ws.iter_rows(min_row=6, values_only=True):
        key = row[0]
        if not key or "|" not in str(key):
            continue
        ds, bb, pid = key.split("|")
        # seeds column
        n_ncl = len(ok[(ok.dataset == ds) & (ok.backbone == bb) & (ok.policy_id == pid) & (ok.objective == "ncl")])
        n_sim = len(ok[(ok.dataset == ds) & (ok.backbone == bb) & (ok.policy_id == pid) & (ok.objective == "simclr")])
        total += 1
        s_ncl = len(ok[(ok.dataset == ds) & (ok.backbone == bb) & (ok.policy_id == pid) & (ok.objective == "ncl") & ok.sc_star.notna()])
        s_sim = len(ok[(ok.dataset == ds) & (ok.backbone == bb) & (ok.policy_id == pid) & (ok.objective == "simclr") & ok.sc_star.notna()])
        want = f"{n_ncl} / {n_sim}" + ("" if (s_ncl, s_sim) == (n_ncl, n_sim) else f" (mono {s_ncl} / {s_sim})")
        if row[2] != want:
            bad += 1; print(f"[seeds] {sheet} {key}: workbook {row[2]!r}, csv {want!r}")
        for j in range(3, len(row)):
            label, obj = labels[j], {"NCL": "ncl", "SimCLR": "simclr"}.get(sub[j])
            if obj is None:
                continue
            exp, n = expected(ds, bb, pid, obj, label)
            got = row[j] if isinstance(row[j], (int, float)) else None
            total += 1
            if (exp is None) != (got is None) or (exp is not None and abs(exp - got) > 1e-6):
                bad += 1; print(f"[value] {sheet} {key} {label} {obj}: workbook {got}, csv {exp} (n={n})")
            elif label in VAL_COL and exp is not None and n != {"SC %": None}.get(label, n) and False:
                print(f"[fewer seeds] {sheet} {key} {label} {obj}: mean of {n} runs, seeds column says "
                      f"{n_ncl if obj == 'ncl' else n_sim}")
print(f"\nsummary cells checked: {total}, mismatches: {bad}")

# ---- Runs (val): every cell against the csv
ws = wb["Runs (val)"]
hdr = [c.value for c in ws[1]]
colmap = {"Probe backbone": "probe_bb_bal_acc", "Probe projector": "probe_proj_bal_acc", "k-NN-20": "knn20_bal_acc",
          "SC*": "sc_star", "SC %": "sc_pct", "MS": "ms", "Purity z*": "purity_proj_star"}
v = val.set_index("run_id")
n_rows = n_bad = 0
for row in ws.iter_rows(min_row=2, values_only=True):
    r = dict(zip(hdr, row)); rid = r["Run id"]
    if rid is None:
        continue
    n_rows += 1
    for h, c in colmap.items():
        e, g = v.loc[rid, c], r[h]
        e = None if (e is None or (isinstance(e, float) and math.isnan(e))) else float(e)
        if (e is None) != (g is None) or (e is not None and abs(e - g) > 1e-6):
            n_bad += 1; print(f"[runs val] {rid} {h}: workbook {g}, csv {e}")
    if (r["Collapsed"] == "yes") != bool(v.loc[rid, "collapsed"]):
        n_bad += 1; print(f"[runs val] {rid} collapsed flag")
print(f"Runs (val): {n_rows} rows (csv has {len(val)}), mismatches: {n_bad}")

ws = wb["Runs (test)"]
hdr = [c.value for c in ws[1]]
tt = t.set_index("run_id"); n_rows = n_bad = 0
for row in ws.iter_rows(min_row=2, values_only=True):
    r = dict(zip(hdr, row)); rid = r["Run id"]
    if rid is None:
        continue
    n_rows += 1
    for h, c in (("Probe backbone", "probe_bb_bal_acc"), ("Probe projector", "probe_proj_bal_acc"), ("k-NN-20", "knn20_bal_acc")):
        if abs(float(tt.loc[rid, c]) - r[h]) > 1e-6:
            n_bad += 1; print(f"[runs test] {rid} {h}: workbook {r[h]}, csv {tt.loc[rid, c]}")
print(f"Runs (test): {n_rows} rows (csv has {len(t)}), mismatches: {n_bad}")
print("collapsed runs left out of means:", val[val.collapsed].run_id.tolist())
print("runs without specialisation scores:", val[val.sc_star.isna()].run_id.tolist())
