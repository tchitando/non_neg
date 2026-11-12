#!/usr/bin/env python3
"""One row per policy and objective: number of seeds, mean and SD of the probe, k-NN and specialisation measures.

    python study_tools/summarise_by_policy.py                                   # ResNet-50 OCT (study/)
    python study_tools/summarise_by_policy.py --study study/rn18 --out study/summary/rn18_val_by_policy.csv
    python study_tools/summarise_by_policy.py --study study/path --out study/summary/path_val_by_policy.csv --n-classes 9

Reads <study>/results_master.csv (probes, k-NN) and <study>/results_spec.csv (SC, SC*, purity, MS, MS* with the readout
used in training; the master's own sc/ms columns are the older readout and are not used). Latest row per run.
SC is the raw class share in % (chance = 100/C), SC* = (SC - 100/C)/(1 - 1/C); MS is Pach et al.'s score on the fixed
balanced subset (no-preference floor in results_spec.csv, ms_floor), MS* = (MS - floor)/(1 - floor).
Validation split only. Untrained-network controls are left out.
"""
import argparse
from pathlib import Path
import pandas as pd

COLS = {"sc_raw": "SC", "ms": "MS", "probe_bb_bal_acc": "probe_backbone", "probe_proj_bal_acc": "probe_projector", "knn5_bal_acc": "knn5",
        "knn20_bal_acc": "knn20", "knn100_bal_acc": "knn100", "sc_star": "SC_star", "purity_proj_star": "purity_projector",
        "purity_bb_star": "purity_backbone", "ms_star": "MS_star", "dead_proj": "dead_units"}
ORDER = ["stem", "contrast", "speckle", "blur", "motion", "pixelate", "jpeg", "rotflip", "stain", "defocus", "cifar",
         "all6", "speckle_contrast_crop020"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="study")
    ap.add_argument("--out", default="study/summary/oct_val_by_policy.csv")
    ap.add_argument("--n-classes", type=int, default=4, help="4 for OCT, 9 for PathMNIST (to recover raw SC from SC*)")
    a = ap.parse_args()
    m = pd.read_csv(Path(a.study) / "results_master.csv").drop_duplicates("run_id", keep="last")
    m = m.drop(columns=[c for c in ("sc", "sc_star", "ms") if c in m.columns])
    s = pd.read_csv(Path(a.study) / "results_spec.csv").drop_duplicates("run_id", keep="last")
    d = m.merge(s[["run_id", "sc_star", "purity_proj_star", "purity_bb_star", "ms", "ms_star"]], on="run_id")
    C = a.n_classes
    d["sc_raw"] = 100 / C + (1 - 1 / C) * d["sc_star"]           # SC in %, undoing the chance correction
    d["objective"] = d.run_id.str.split("-").str[0]
    g = d.groupby(["policy_id", "objective"])
    out = g.size().rename("n_seeds").to_frame()
    out["seeds"] = g["seed"].apply(lambda x: " ".join(str(int(v)) for v in sorted(x)))
    for c, n in COLS.items():
        if c in d.columns:
            out[n + "_mean"] = g[c].mean()
            out[n + "_sd"] = g[c].std(ddof=1)
    out = out.reset_index()
    out["_k"] = out.policy_id.map(lambda p: (ORDER.index(p) if p in ORDER else len(ORDER), p))
    out = out.sort_values(["_k", "objective"]).drop(columns="_k")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(a.out, index=False)
    print(f"{len(out)} rows -> {a.out}")
    show = ["policy_id", "objective", "n_seeds", "probe_backbone_mean", "probe_backbone_sd", "knn20_mean",
            "SC_mean", "SC_star_mean", "MS_mean", "MS_star_mean", "purity_projector_mean"]
    with pd.option_context("display.width", 250):
        print(out[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
