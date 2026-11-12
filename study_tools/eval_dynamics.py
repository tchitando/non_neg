#!/usr/bin/env python3
"""Score every saved checkpoint of the training-dynamics runs (validation split only).

    CUDA_VISIBLE_DEVICES=0 python study_tools/eval_dynamics.py              # all runs in study/dyn/manifest.csv
    CUDA_VISIBLE_DEVICES=0 python study_tools/eval_dynamics.py --list       # which checkpoints exist / are scored
    CUDA_VISIBLE_DEVICES=0 python study_tools/eval_dynamics.py --only '^ncl-'

Per checkpoint (epochs 0, 9, ..., 99), with the same code and settings as the paper's validation scores:
    probe_bb_bal_acc, probe_proj_bal_acc   eval_run.linear_probe (LARS, 50 epochs) on h and on z (NCL: after ReLU)
    knn20_bal_acc                          eval_run.knn_bal_acc, k = 20, on h
    sc_star, sc_pct, purity_proj_star      eval_spec.audit on 5 class-balanced validation draws, training readout
    alive_units                            projector units active on at least one validation image
MS is not computed (it needs the DINOv2 reference and is not part of the dynamics figure).
A checkpoint already in --out is never scored again, so the script can be rerun while training is still going.
"""
import argparse, csv, glob, os, re, sys, time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
FIELDS = ["run_id", "objective", "policy_id", "seed", "epoch", "ckpt_path", "probe_bb_bal_acc", "probe_proj_bal_acc",
          "knn20_bal_acc", "sc_star", "sc_pct", "purity_proj_star", "alive_units", "eval_date"]


def checkpoints(run_id, root):
    hits = glob.glob(os.path.join(root, "**", f"{run_id}-offline-*-ep=*.ckpt"), recursive=True)
    out = {}
    for h in hits:
        m = re.search(r"-ep=(\d+)\.ckpt$", h)
        if m:
            out[int(m.group(1))] = h
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="study/dyn/manifest.csv")
    ap.add_argument("--out", default="study/dyn/dyn_results.csv")
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--only", default=".")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE)); sys.path.insert(0, ".")
    import eval_run as ER
    import eval_spec as ES
    ER.set_dataset("octmnist")
    C = ER.N_CLASSES

    runs = [r for r in csv.DictReader(open(a.manifest)) if re.search(a.only, r["run_id"])]
    out = Path(a.out)
    done = {(r["run_id"], int(r["epoch"])) for r in csv.DictReader(open(out))} if out.exists() else set()
    todo = []
    for r in runs:
        ck = checkpoints(r["run_id"], a.ckpt_root)
        n_done = sum((r["run_id"], e) in done for e in ck)
        print(f"{r['run_id']:44s} {len(ck):2d} checkpoints, {n_done} scored  epochs {list(ck)}")
        todo += [(r, e, p) for e, p in ck.items() if (r["run_id"], e) not in done]
    print(f"\n{len(todo)} checkpoint(s) to score")
    if a.list or not todo:
        return

    new = not out.exists()
    for r, ep, path in todo:
        t0 = time.time()
        ncl = r["objective"] == "ncl"
        bb, pj, _ = ER.load_model(path)
        htr, ztr_raw, ytr = ER.extract(bb, pj, "train", False, a.bs, a.workers)
        hva, zva_raw, yva = ER.extract(bb, pj, "val", False, a.bs, a.workers)
        ztr = F.relu(ztr_raw) if ncl else ztr_raw
        zva = F.relu(zva_raw) if ncl else zva_raw
        p_bb = ER.linear_probe(htr, ytr, hva, yva)
        p_pj = ER.linear_probe(ztr, ytr, zva, yva)
        knn = ER.knn_bal_acc(htr, ytr, hva, yva, ks=(20,))
        spec = ES.audit(hva, ES.readout(zva_raw, ncl), yva, ncl, C)
        alive = int(((zva.sum(0) > 0) if ncl else (zva.abs().sum(0) > 0)).sum())
        row = dict(run_id=r["run_id"], objective=r["objective"], policy_id=r["policy_id"], seed=r["seed"], epoch=ep,
                   ckpt_path=path, probe_bb_bal_acc=round(p_bb[0], 3), probe_proj_bal_acc=round(p_pj[0], 3),
                   knn20_bal_acc=round(knn[20], 3), sc_star=round(spec["sc_star"], 3),
                   sc_pct=round(100 / C + (1 - 1 / C) * spec["sc_star"], 3),
                   purity_proj_star=round(spec["purity_proj_star"], 4), alive_units=alive,
                   eval_date=datetime.now().strftime("%Y-%m-%d %H:%M"))
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader(); new = False
            w.writerow(row)
        print(f"{r['run_id']} ep {ep:3d}: probe z {row['probe_proj_bal_acc']}  probe h {row['probe_bb_bal_acc']}  "
              f"SC {row['sc_pct']}%  purity {row['purity_proj_star']}  alive {alive}  ({time.time() - t0:.0f}s)", flush=True)
        del bb, pj, htr, ztr, hva, zva, ztr_raw, zva_raw
        torch.cuda.empty_cache()
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
