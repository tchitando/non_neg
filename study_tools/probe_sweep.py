#!/usr/bin/env python3
"""Linear-probe hyperparameter check: does the fixed probe recipe treat NCL and SimCLR fairly?

    CUDA_VISIBLE_DEVICES=0 python study_tools/probe_sweep.py                       # 4 checkpoints x 2 layers (default)
    CUDA_VISIBLE_DEVICES=0 python study_tools/probe_sweep.py --only '^(ncl|simclr)-resnet50-octmnist-(stem|contrast)(-s4[34])?$'

Every run is probed today with one recipe: LARS, lr 0.1, weight decay 1e-4, 50 epochs (10 warm-up, cosine), batch
256, raw features (backbone h; projector z after ReLU for NCL, signed z for SimCLR). This script, per checkpoint and
layer:
  1. extracts train, validation and test features once;
  2. splits the training images 90 / 10 (stratified, seed 0); the 10% is the selection set, validation is not used;
  3. trains a probe for every setting in the grid on the 90% and scores it on the 10% (balanced accuracy):
        lr {0.01, 0.03, 0.1, 0.3, 1.0} x weight decay {0, 1e-6, 1e-4} x standardisation {off, on}
     standardisation = each feature dimension z-scored with training-set mean and SD (a BatchNorm without learned
     scale, as in MAE's linear probe);
  4. refits the best setting on all training images and reports validation and test balanced accuracy, next to the
     fixed recipe refit the same way.
Rows go to study/derived/probe_sweep.csv (one per setting) and study/derived/probe_sweep_summary.csv (one per run and
layer); finished settings are skipped on a rerun. The printed summary ends with the NCL - SimCLR gaps under both
recipes: if the tuned gaps and the policy ranking match the fixed ones, the fixed recipe stands.
"""
import argparse, csv, math, re, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

HERE = Path(__file__).resolve().parent
LRS = [0.01, 0.03, 0.1, 0.3, 1.0]
WDS = [0.0, 1e-6, 1e-4]
STDS = [False, True]
FIXED = (0.1, 1e-4, False)
DEFAULT_ONLY = r"^(ncl|simclr)-resnet50-octmnist-(stem|contrast)$"
GRID_FIELDS = ["run_id", "objective", "policy_id", "seed", "layer", "lr", "wd", "std", "select_bal_acc", "eval_date"]
SUM_FIELDS = ["run_id", "objective", "policy_id", "seed", "layer", "fixed_select", "fixed_val", "fixed_test",
              "best_lr", "best_wd", "best_std", "best_select", "tuned_val", "tuned_test", "eval_date"]


def make_opt(params, lr, wd):
    try:
        from solo.utils.lars import LARS
        return LARS(params, lr=lr, weight_decay=wd, clip_lr=True, eta=0.02, exclude_bias_n_norm=True)
    except Exception:
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)


def train_probe(X, y, n_classes, lr, wd, epochs=50, warmup=10, bs=256, seed=0):
    """The fixed recipe of eval_run.linear_probe with lr and wd as arguments; returns the trained layer."""
    torch.manual_seed(seed)
    clf = nn.Linear(X.shape[1], n_classes).cuda()
    opt = make_opt(clf.parameters(), lr, wd)
    spe = math.ceil(len(X) / bs)
    total, warm = epochs * spe, warmup * spe
    lam = lambda s: s / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lam)
    Xg, yg = X.cuda(), y.cuda()
    for _ in range(epochs):
        perm = torch.randperm(len(Xg), device=Xg.device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(Xg[idx]), yg[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return clf.eval()


@torch.no_grad()
def bal_acc(clf, X, y):
    pred = torch.cat([clf(X[i:i + 8192].cuda()).argmax(1).cpu() for i in range(0, len(X), 8192)])
    return 100 * balanced_accuracy_score(y.numpy(), pred.numpy())


def standardise(Xtr, *others):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    return [(Xtr - mu) / sd] + [(X - mu) / sd for X in others]


def strat_split(y, frac=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    sel = []
    for c in torch.unique(y):
        idx = (y == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        sel.append(idx[: max(1, int(round(frac * len(idx))))])
    sel = torch.cat(sel)
    mask = torch.ones(len(y), dtype=torch.bool); mask[sel] = False
    return mask.nonzero(as_tuple=True)[0], sel


def append(path, fields, row):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=DEFAULT_ONLY, help="regex on run ids in study/results_master.csv")
    ap.add_argument("--master", default="study/results_master.csv")
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--out", default="study/derived/probe_sweep.csv")
    ap.add_argument("--summary", default="study/derived/probe_sweep_summary.csv")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sys.path.insert(0, str(HERE)); sys.path.insert(0, ".")
    import eval_run as ER
    ER.set_dataset("octmnist")
    C = ER.N_CLASSES

    rows = {r["run_id"]: r for r in csv.DictReader(open(a.master)) if re.search(a.only, r["run_id"])}
    if not rows:
        sys.exit(f"no run in {a.master} matches {a.only}")
    out, summ = Path(a.out), Path(a.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["run_id"], r["layer"], float(r["lr"]), float(r["wd"]), r["std"] == "True")
            for r in csv.DictReader(open(out))} if out.exists() else set()
    summ_done = {(r["run_id"], r["layer"]) for r in csv.DictReader(open(summ))} if summ.exists() else set()
    grid = [(lr, wd, sd) for sd in STDS for wd in WDS for lr in LRS]
    print(f"{len(rows)} runs x 2 layers x {len(grid)} settings; selection on 10% of the training images")

    for rid, r in sorted(rows.items()):
        if all((rid, lay) in summ_done for lay in ("backbone", "projector")):
            print(f"{rid}: done"); continue
        hits = ER.find_ckpts(rid, a.ckpt_root)
        if not hits:
            print(f"{rid}: no checkpoint"); continue
        t0 = time.time()
        ncl = r["objective"] == "ncl"
        bb, pj, _ = ER.load_model(hits[-1])
        feats = {}
        for split in ("train", "val", "test"):
            h, z, y = ER.extract(bb, pj, split, False, a.bs, a.workers)
            feats[split] = (h, F.relu(z) if ncl else z, y.long())
        del bb, pj; torch.cuda.empty_cache()
        print(f"{rid}: features extracted ({time.time() - t0:.0f}s)", flush=True)
        ytr = feats["train"][2]
        fit_idx, sel_idx = strat_split(ytr)
        for li, layer in enumerate(("backbone", "projector")):
            if (rid, layer) in summ_done:
                continue
            Xtr, Xva, Xte = feats["train"][li], feats["val"][li], feats["test"][li]
            yva, yte = feats["val"][2], feats["test"][2]
            Xtr_s, Xva_s, Xte_s = standardise(Xtr, Xva, Xte)
            Xfit_s, Xsel_s = standardise(Xtr[fit_idx], Xtr[sel_idx])
            scores = {}
            for lr, wd, sd in grid:
                key = (rid, layer, lr, wd, sd)
                Xf, Xs = (Xfit_s, Xsel_s) if sd else (Xtr[fit_idx], Xtr[sel_idx])
                if key in done:
                    continue
                clf = train_probe(Xf, ytr[fit_idx], C, lr, wd)
                s = bal_acc(clf, Xs, ytr[sel_idx])
                append(out, GRID_FIELDS, dict(run_id=rid, objective=r["objective"], policy_id=r["policy_id"], seed=r["seed"],
                                              layer=layer, lr=lr, wd=wd, std=sd, select_bal_acc=round(s, 3),
                                              eval_date=datetime.now().strftime("%Y-%m-%d %H:%M")))
                done.add(key)
            for g in csv.DictReader(open(out)):
                if g["run_id"] == rid and g["layer"] == layer:
                    scores[(float(g["lr"]), float(g["wd"]), g["std"] == "True")] = float(g["select_bal_acc"])
            best = max(scores, key=scores.get)
            res = {}
            for name, (lr, wd, sd) in (("fixed", FIXED), ("tuned", best)):
                Xa, Xb, Xc = (Xtr_s, Xva_s, Xte_s) if sd else (Xtr, Xva, Xte)
                clf = train_probe(Xa, ytr, C, lr, wd)
                res[name] = (bal_acc(clf, Xb, yva), bal_acc(clf, Xc, yte))
            row = dict(run_id=rid, objective=r["objective"], policy_id=r["policy_id"], seed=r["seed"], layer=layer,
                       fixed_select=round(scores[FIXED], 3), fixed_val=round(res["fixed"][0], 3), fixed_test=round(res["fixed"][1], 3),
                       best_lr=best[0], best_wd=best[1], best_std=best[2], best_select=round(scores[best], 3),
                       tuned_val=round(res["tuned"][0], 3), tuned_test=round(res["tuned"][1], 3),
                       eval_date=datetime.now().strftime("%Y-%m-%d %H:%M"))
            append(summ, SUM_FIELDS, row)
            print(f"  {layer:9s} fixed val {row['fixed_val']:6.2f} test {row['fixed_test']:6.2f} | best lr {best[0]} wd {best[1]} "
                  f"std {best[2]} -> val {row['tuned_val']:6.2f} test {row['tuned_test']:6.2f}  ({time.time() - t0:.0f}s)", flush=True)

    import pandas as pd
    s = pd.read_csv(summ)
    s = s[s.run_id.apply(lambda x: re.search(a.only, x) is not None)]
    print("\nSummary (balanced accuracy, %): fixed recipe vs tuned on the 10% selection split")
    print(s[["run_id", "layer", "fixed_val", "tuned_val", "fixed_test", "tuned_test", "best_lr", "best_wd", "best_std"]]
          .to_string(index=False))
    g = s.groupby(["policy_id", "layer", "objective"])[["fixed_val", "tuned_val", "fixed_test", "tuned_test"]].mean().unstack("objective")
    if ("fixed_val", "ncl") in g and ("fixed_val", "simclr") in g:
        print("\nNCL - SimCLR, fixed vs tuned")
        for (pol, lay), v in g.iterrows():
            d = {k: v[(k, "ncl")] - v[(k, "simclr")] for k in ("fixed_val", "tuned_val", "fixed_test", "tuned_test")}
            print(f"  {pol:10s} {lay:9s} val {d['fixed_val']:+6.2f} -> {d['tuned_val']:+6.2f}   test {d['fixed_test']:+6.2f} -> {d['tuned_test']:+6.2f}")


if __name__ == "__main__":
    main()
