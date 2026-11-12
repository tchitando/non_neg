#!/usr/bin/env python3
"""Semantic consistency (SC), the reference implementation for the paper.

SC (Wang et al., ICLR 2024, "class consistency"; used by Zhang et al., ICLR 2025, as "semantic consistency"):
for each projector unit that is ever active, take the images on which it fires (|z| > 1e-5 after each image's
feature vector is L2-normalised), find the most frequent class among them, and record that class's share.
SC is the mean share over units, in %. Chance = 100 / number of classes on a class-balanced set.

This file holds:
    ncl_cluster_acc   NCL's own function (original_non_neg/main_eval.py, cluster_acc), copied line for line
                      apart from the print statements, so our numbers can be checked against theirs.
    sc_percent        the vectorised version used in eval_spec.py and eval_test.py (identical result).
    readout           the projector output as the loss sees it (what both functions are applied to).

Usage
    python study_tools/sc_reference.py --selftest
        checks that sc_percent and ncl_cluster_acc give the same SC on random data (non-negative and signed).
    CUDA_VISIBLE_DEVICES=0 python study_tools/sc_reference.py --dataset octmnist --only 'resnet50-octmnist-stem'
        raw SC (%) for the matching runs (add --master study/rn18/results_master.csv for ResNet-18), two ways:
          full_val      NCL's function on the whole validation split (NCL's protocol: their eval split)
          balanced_val  our protocol: mean over 5 class-balanced subsets of the validation split (as eval_spec.py)
        The validation split of OCTMNIST is imbalanced (about 47% Normal), so full_val is not comparable with the
        25% chance level; balanced_val is the number that goes in the paper.

Differences from NCL's code, all deliberate:
    1. Data: class-balanced validation subsets (5 draws, seed 0-4) instead of a whole evaluation split, so that
       chance is exactly 1/C on imbalanced medical datasets. On a balanced split the two coincide up to sampling.
    2. Live-unit mask: NCL uses relu(z).sum(0) > 0 for every model; for SimCLR's signed output we use
       |z|.sum(0) > 0. For non-negative NCL output the two masks are identical; for SimCLR they differ only for a
       unit that is never positive, and SimCLR's SC sits at chance either way.
    3. Readout: NCL applies ReLU to the raw projector output and then L2-normalises; we L2-normalise, apply ReLU,
       and L2-normalise again (the training-time readout). Normalising by a positive number before the ReLU does
       not change which entries are zero or their direction, so the result is the same.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EPS = 1e-5


def ncl_cluster_acc(features, labels, eps=EPS, take_abs=False, topk=False):
    """original_non_neg/main_eval.py, cluster_acc (Wang et al. 2024), without the prints. Returns SC per unit, in %."""
    features = features[:, F.relu(features).sum(0) > 0]
    features = F.normalize(features, dim=1)
    if take_abs:
        features = features.abs()
    if topk:
        sorted_, indices = torch.sort(features.sum(dim=0), descending=True)
        indices = indices[sorted_ > 1]
        features = features[:, indices]
    acc_per_dim = []
    for i in range(features.shape[1]):
        mask = features.abs()[:, i] > eps
        labels_selected = labels[mask]
        try:
            dist = labels_selected.bincount()
            dist = dist / dist.sum()
            acc_per_dim.append(dist.max().item())
        except Exception:
            pass
    return (torch.tensor(acc_per_dim) * 100).numpy()


def sc_percent(Zb, yb, relu, n_classes, eps=EPS):
    """Vectorised SC in % (the function in eval_spec.py). relu=True for NCL (non-negative readout)."""
    alive = (Zb.sum(0) > 0) if relu else (Zb.abs().sum(0) > 0)
    if int(alive.sum()) == 0:
        return float("nan")
    Zn = F.normalize(Zb[:, alive], dim=1)
    fire = (Zn.abs() > eps).float()
    onehot = F.one_hot(yb, n_classes).float()
    counts = onehot.T @ fire
    tot = counts.sum(0)
    ok = tot > 0
    return 100 * (counts.max(0).values[ok] / tot[ok]).mean().item()


def readout(z_raw, ncl):
    """Projector output as the contrastive loss sees it: SimCLR L2-normalised; NCL normalise -> ReLU -> normalise."""
    z = F.normalize(z_raw, dim=1)
    return F.normalize(F.relu(z), dim=1) if ncl else z


def balanced_index(y, seed, n_classes):
    g = torch.Generator().manual_seed(seed)
    n = min(int((y == c).sum()) for c in range(n_classes))
    return torch.cat([(y == c).nonzero(as_tuple=True)[0][torch.randperm(int((y == c).sum()), generator=g)[:n]]
                      for c in range(n_classes)])


def selftest():
    torch.manual_seed(0)
    worst = 0.0
    for trial in range(20):
        n, d, C = 600, 64, 4
        y = torch.randint(0, C, (n,))
        z = torch.randn(n, d)
        z[:, : d // 2] += 2.0 * F.one_hot(y, C).float().repeat(1, d // (2 * C))   # some class-tied units
        z[:, -3:] = 0                                                                # dead units
        for ncl in (True, False):
            r = readout(z, ncl)
            a = sc_percent(r, y, ncl, C)
            b = float(np.mean(ncl_cluster_acc(r if ncl else r, y)))
            if not ncl:
                # NCL's live mask for signed output is relu(z).sum > 0; ours is |z|.sum > 0. Compare on the units both keep.
                keep = (F.relu(r).sum(0) > 0) & (r.abs().sum(0) > 0)
                a = sc_percent(r[:, keep], y, False, C)
            worst = max(worst, abs(a - b))
        # NCL's offline pipeline: ReLU on the raw output, then their function normalises; ours: readout
        b_raw = float(np.mean(ncl_cluster_acc(F.relu(z), y)))
        a_raw = sc_percent(readout(z, True), y, True, C)
        worst = max(worst, abs(a_raw - b_raw))
    print(f"selftest: largest difference between sc_percent and NCL's cluster_acc over 20 random cases: {worst:.2e} %")
    assert worst < 1e-4, "the two implementations disagree"
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", default="octmnist")
    ap.add_argument("--only", default=".", help="regex on run_id (runs listed in the dataset's results_master.csv)")
    ap.add_argument("--master", default=None, help="results_master.csv listing the runs (default: the dataset's; "
                    "study/rn18/results_master.csv for ResNet-18)")
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="optional csv to append results to")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    import csv, re
    sys.path.insert(0, str(HERE)); sys.path.insert(0, ".")
    import eval_run as ER
    ER.set_dataset(a.dataset)
    C = ER.N_CLASSES
    master = Path(a.master or ("study/results_master.csv" if a.dataset == "octmnist" else "study/path/results_master.csv"))
    runs = [r["run_id"] for r in csv.DictReader(open(master)) if re.search(a.only, r["run_id"])]
    runs = list(dict.fromkeys(runs))
    rows = []
    for rid in runs:
        hits = ER.find_ckpts(rid, a.ckpt_root)
        if not hits:
            print(f"{rid}: no checkpoint"); continue
        ncl = rid.startswith("ncl")
        bb, pj, _ = ER.load_model(hits[-1])
        _, z_raw, y = ER.extract(bb, pj, "val", False, a.bs, a.workers)
        z = readout(z_raw, ncl)
        full = float(np.mean(ncl_cluster_acc(F.relu(z_raw) if ncl else z_raw, y)))
        bal = [sc_percent(z[idx], y[idx], ncl, C) for idx in (balanced_index(y, s, C) for s in range(a.draws))]
        bal = float(np.nanmean(bal))
        row = dict(run_id=rid, sc_full_val=round(full, 3), sc_balanced_val=round(bal, 3), chance=round(100 / C, 2))
        rows.append(row)
        print(f"{rid:58s} SC full val {full:6.2f}%   SC balanced val {bal:6.2f}%   (chance {100 / C:.1f}%)", flush=True)
        del bb, pj
        torch.cuda.empty_cache()
    if a.out and rows:
        new = not Path(a.out).exists()
        with open(a.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
