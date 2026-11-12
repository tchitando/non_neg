#!/usr/bin/env python3
"""Derived tables from the master results table (study/results_master.csv). Reads only; writes study/derived/.

    python study_tools/summarise.py                 # everything in the master table
    python study_tools/summarise.py --policies stem speckle contrast blur motion pixelate jpeg cifar

Outputs
    derived/stem_noise.csv   stem seeds per objective and metric: mean, sd, min, max, range
    derived/effects.csv      for each policy and objective: value, difference from the stem mean, and
                             'outside_stem_range' (the value lies outside the min..max of the stem seeds)
    derived/rank_agreement.csv  Spearman correlation between the probe ranking and the specialisation ranking (NCL)
The latest row per run_id is used (the master table is append-only; a re-score adds a later row).
"""
import argparse, csv, math, statistics as st
from pathlib import Path

METRICS = ["probe_bb_bal_acc", "probe_proj_bal_acc", "knn20_bal_acc", "sc_star", "ms", "effdim_proj", "alive_proj"]


def load(path):
    latest = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            latest[r["run_id"]] = r          # later rows replace earlier ones
    return list(latest.values())


def num(x):
    try:
        return float(x)
    except Exception:
        return None


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num_ = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num_ / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--out", default="study/derived")
    ap.add_argument("--policies", nargs="*", default=None)
    a = ap.parse_args()
    rows = load(a.master)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    by = {}   # (objective, policy) -> list of rows (seeds)
    for r in rows:
        by.setdefault((r["objective"], r["policy_id"]), []).append(r)

    noise, effects = [], []
    for obj in ("ncl", "simclr"):
        stem = by.get((obj, "stem"), [])
        if not stem:
            continue
        stats = {}
        for m in METRICS:
            v = [num(r[m]) for r in stem if num(r[m]) is not None]
            if not v:
                continue
            stats[m] = dict(mean=st.mean(v), sd=st.stdev(v) if len(v) > 1 else float("nan"), lo=min(v), hi=max(v))
            noise.append(dict(objective=obj, metric=m, n_seeds=len(v), mean=round(stats[m]["mean"], 4),
                              sd=round(stats[m]["sd"], 4), min=round(min(v), 4), max=round(max(v), 4),
                              range=round(max(v) - min(v), 4)))
        pols = sorted({p for (o, p) in by if o == obj})
        if a.policies:
            pols = [p for p in a.policies if p in pols]
        for p in pols:
            rs = by[(obj, p)]
            row = dict(objective=obj, policy=p, n_seeds=len(rs), augs_on=rs[0]["augs_on"])
            for m, s in stats.items():
                v = [num(r[m]) for r in rs if num(r[m]) is not None]
                if not v:
                    continue
                x = st.mean(v)
                row[m] = round(x, 4)
                row[m + "_diff"] = round(x - s["mean"], 4)
                row[m + "_outside_stem"] = "" if p == "stem" else ("above" if x > s["hi"] else "below" if x < s["lo"] else "")
            effects.append(row)

    def write(name, recs):
        if not recs:
            return
        keys = []
        for r in recs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(recs)

    write("stem_noise.csv", noise)
    write("effects.csv", effects)

    # agreement between the probe ranking and the specialisation ranking (policies other than stem seeds averaged)
    agree = []
    for obj in ("ncl",):
        recs = [r for r in effects if r["objective"] == obj and "sc_star" in r and "probe_bb_bal_acc" in r]
        if len(recs) >= 4:
            agree.append(dict(objective=obj, n_policies=len(recs),
                              spearman_probe_vs_sc_star=round(spearman([r["probe_bb_bal_acc"] for r in recs], [r["sc_star"] for r in recs]), 3),
                              spearman_probe_vs_ms=round(spearman([r["probe_bb_bal_acc"] for r in recs], [r["ms"] for r in recs]), 3) if all("ms" in r for r in recs) else ""))
    write("rank_agreement.csv", agree)

    # console view
    print("\nSTEM NOISE (seeds):")
    for n in noise:
        if n["metric"] in ("probe_bb_bal_acc", "sc_star", "ms", "knn20_bal_acc"):
            print(f"  {n['objective']:6s} {n['metric']:18s} mean {n['mean']:9.4f}  sd {n['sd']:7.4f}  range {n['min']:.4f}..{n['max']:.4f}")
    print("\nEFFECTS (difference from stem mean; * = outside the stem seeds' min..max)")
    hdr = f"{'objective':7s}{'policy':10s}" + "".join(f"{m.replace('_bal_acc','').replace('probe_','p_'):>12s}" for m in ("probe_bb_bal_acc", "probe_proj_bal_acc", "knn20_bal_acc", "sc_star", "ms", "effdim_proj"))
    print(hdr)
    for r in effects:
        line = f"{r['objective']:7s}{r['policy']:10s}"
        for m in ("probe_bb_bal_acc", "probe_proj_bal_acc", "knn20_bal_acc", "sc_star", "ms", "effdim_proj"):
            if m + "_diff" in r:
                d = r[m + "_diff"]; flag = "*" if r.get(m + "_outside_stem") else " "
                line += f"{d:+11.3f}{flag}"
            else:
                line += f"{'':>12s}"
        print(line)
    for g in agree:
        print("\nrank agreement:", g)
    print(f"\nwrote {out}/stem_noise.csv, effects.csv, rank_agreement.csv")


if __name__ == "__main__":
    main()
