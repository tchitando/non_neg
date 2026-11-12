#!/usr/bin/env python3
"""Test-split evaluation: the only script in the study that reads the test split.

It scores exactly the runs listed in a test plan that was written before any test result was seen, and nothing else:

    CUDA_VISIBLE_DEVICES=0 python study_tools/eval_test.py --plan study/test_plan_oct.csv
    CUDA_VISIBLE_DEVICES=0 python study_tools/eval_test.py --plan study/test_plan_oct.csv --list   # status only

Runs whose epoch-99 checkpoint does not exist yet are skipped; run the same command later to add them. A run that is
already in the results file is never scored again, so every test number is computed once.

Per run (the probe is trained on the full training split with the same recipe and probe seed as eval_run.py):
    probe_bb_*     linear probe on the backbone feature h: balanced accuracy, accuracy, macro one-vs-rest AUC, and
                   recall per class (recall_bb, a JSON list in class order)
    probe_proj_*   the same on the projector output (after the ReLU for NCL, as in eval_run.py)
    knn20_bal_acc  k-NN (k = 20, cosine, training set as the bank) on h
    sc_star, purity_proj_star, purity_bb_star
                   eval_spec.py's measures on the class-balanced test set, with z read out as in training
Results are appended to --out (default: study/test_results_<dataset>.csv).
"""
import argparse, csv, hashlib, json, math, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent

FIELDS = [
    "run_id", "dataset", "backbone", "objective", "seed", "policy_id", "ckpt_path", "split", "n_test",
    "probe_bb_bal_acc", "probe_bb_acc", "probe_bb_auc", "recall_bb",
    "probe_proj_bal_acc", "probe_proj_acc", "probe_proj_auc", "recall_proj",
    "knn20_bal_acc", "sc_star", "sc_star_shuf", "purity_proj_star", "purity_bb_star",
    "plan", "eval_date", "script_md5",
]


def probe_fit(Xtr, ytr, n_classes, seed=0, epochs=50, lr=0.1, wd=1e-4, warmup=10, bs=256, device="cuda"):
    """The linear probe of eval_run.linear_probe (LARS, 10 warm-up epochs, cosine, 50 epochs, last epoch), returning
    the trained classifier so that test logits, AUC and per-class recall can be computed."""
    torch.manual_seed(seed); np.random.seed(seed)
    clf = nn.Linear(Xtr.shape[1], n_classes).to(device)
    try:
        from solo.utils.lars import LARS
        opt = LARS(clf.parameters(), lr=lr, weight_decay=wd, clip_lr=True, eta=0.02, exclude_bias_n_norm=True)
        opt_name = "LARS"
    except Exception:
        opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        opt_name = "SGD(fallback)"
    spe = math.ceil(len(Xtr) / bs)
    total, warm = epochs * spe, warmup * spe

    def lam(step):
        if step < warm:
            return step / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lam)
    Xg, yg = Xtr.to(device), ytr.to(device)
    for _ in range(epochs):
        clf.train()
        perm = torch.randperm(len(Xg), device=Xg.device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(Xg[idx]), yg[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    clf.eval()
    return clf, opt_name


def scores(logits, y, n_classes):
    """Balanced accuracy, accuracy, macro one-vs-rest AUC (%), and recall per class (%)."""
    p = torch.softmax(logits, 1).numpy()
    pred = p.argmax(1)
    y = y.numpy()
    labels = list(range(n_classes))
    bal = 100 * balanced_accuracy_score(y, pred)
    acc = 100 * float((pred == y).mean())
    auc = 100 * roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels) if n_classes > 2 \
        else 100 * roc_auc_score(y, p[:, 1])
    rec = (100 * recall_score(y, pred, average=None, labels=labels)).round(2).tolist()
    return bal, acc, auc, rec


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="test plan csv (run_id, dataset, backbone, objective, seed, policy_id)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="only show which planned runs are trained / scored")
    a = ap.parse_args()

    sys.path.insert(0, str(HERE)); sys.path.insert(0, ".")
    import eval_run as ER
    import eval_spec as ES

    plan = read_csv(a.plan)
    datasets = {r["dataset"] for r in plan}
    if len(datasets) != 1:
        sys.exit(f"one dataset per plan, found {sorted(datasets)}")
    ds = datasets.pop()
    ER.set_dataset(ds)
    out = Path(a.out or f"study/test_results_{ds}.csv")
    done = {r["run_id"] for r in read_csv(out)} if out.exists() else set()

    todo = []
    for r in plan:
        hits = ER.find_ckpts(r["run_id"], a.ckpt_root)
        status = "scored" if r["run_id"] in done else ("not trained yet" if not hits else "ready")
        print(f"{r['run_id']:58s} {status}")
        if status == "ready":
            r["_ckpt"] = hits[-1]
            todo.append(r)
    n_tr = sum(1 for r in plan if ER.find_ckpts(r["run_id"], a.ckpt_root))
    print(f"\nplan {a.plan}: {len(plan)} runs, {n_tr} trained, {len(done & {r['run_id'] for r in plan})} scored, "
          f"{len(todo)} to score now")
    if a.list or not todo:
        return

    md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12]
    C = ER.N_CLASSES
    new = not out.exists()
    for r in todo:
        t0 = time.time()
        ncl = r["objective"] == "ncl"
        print(f"=== {r['run_id']}  {r['_ckpt']}", flush=True)
        bb, pj, _ = ER.load_model(r["_ckpt"])
        htr, ztr_raw, ytr = ER.extract(bb, pj, "train", False, a.bs, a.workers)
        hte, zte_raw, yte = ER.extract(bb, pj, "test", False, a.bs, a.workers)
        ztr = F.relu(ztr_raw) if ncl else ztr_raw          # the probe input of eval_run.py
        zte = F.relu(zte_raw) if ncl else zte_raw
        res = {}
        for name, Xtr, Xte in (("bb", htr, hte), ("proj", ztr, zte)):
            clf, opt_name = probe_fit(Xtr, ytr, C)
            with torch.no_grad():
                logits = clf(Xte.cuda()).cpu()
            bal, acc, auc, rec = scores(logits, yte, C)
            res.update({f"probe_{name}_bal_acc": round(bal, 3), f"probe_{name}_acc": round(acc, 3),
                        f"probe_{name}_auc": round(auc, 3), f"recall_{name}": json.dumps(rec)})
        knn = ER.knn_bal_acc(htr, ytr, hte, yte, ks=(20,))
        spec = ES.audit(hte, ES.readout(zte_raw, ncl), yte, ncl, C)
        row = dict(run_id=r["run_id"], dataset=ds, backbone=r.get("backbone", ""), objective=r["objective"],
                   seed=r["seed"], policy_id=r["policy_id"], ckpt_path=r["_ckpt"], split="test", n_test=len(yte),
                   knn20_bal_acc=round(knn[20], 3), sc_star=round(spec["sc_star"], 3),
                   sc_star_shuf=round(spec["sc_star_shuf"], 3), purity_proj_star=round(spec["purity_proj_star"], 4),
                   purity_bb_star=round(spec["purity_bb_star"], 4), plan=a.plan,
                   eval_date=datetime.now().strftime("%Y-%m-%d %H:%M"), script_md5=md5, **res)
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader(); new = False
            w.writerow(row)
        print(f"    probe(bb) {row['probe_bb_bal_acc']}  probe(proj) {row['probe_proj_bal_acc']}  "
              f"AUC {row['probe_bb_auc']}  knn20 {row['knn20_bal_acc']}  SC* {row['sc_star']}  "
              f"purity z {row['purity_proj_star']}  recall {row['recall_bb']}  ({time.time() - t0:.0f}s)", flush=True)
        del bb, pj, htr, ztr, hte, zte, ztr_raw, zte_raw
        torch.cuda.empty_cache()
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
