#!/usr/bin/env python3
"""Label-preservation test for COMBINATIONS: does the diagnosis survive when several augmentations are stacked?

Six augmentations that each pass the single-augmentation test can still destroy the scan together. For each
stack this trains one fresh ResNet-18 on training images that carry the stack (every augmentation applied
independently with probability `prob` from study/levels.json, in the order the pretraining pipeline uses),
then scores it on validation images:
    clean        no augmentation
    drawn        the stack as pretraining sees it (each augmentation with probability prob)
    forced       every augmentation in the stack applied to every image (worst case)
and compares with the clean baseline (study/label_test/clf.pt on clean validation images).

    python study_tools/label_test_stack.py                          # the full six-way stack
    python study_tools/label_test_stack.py --all-fives              # + the six five-way (leave-one-out) stacks
    python study_tools/label_test_stack.py --stacks speckle_contrast_blur contrast_motion

A stack passes when the DRAWN balanced-accuracy drop is at most --tolerance points and no class loses more than
3x --tolerance. The forced column is reported for information. The crop and brightness jitter of the fixed stem
are not part of this test, as in the single-augmentation test. Results go to <out>/label_test_stack.csv and summary.txt
after each stack. Validation split only.
"""
import argparse, csv, json, random
from pathlib import Path

from aug_ladders import POOL, from_levels

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, recall_score

from label_test import NpzDS, make_model, predict, train


class Stack:
    def __init__(self, fns, p, force=False):
        self.fns, self.p, self.force = fns, p, force

    def __call__(self, img):
        for f in self.fns:
            if self.force or random.random() < self.p:
                img = f(img)
        return img


def parse(s):
    names = [] if s == "stem" else (POOL if s == "all6" else s.split("_"))
    bad = [n for n in names if n not in POOL]
    if bad:
        raise SystemExit(f"unknown augmentation {bad} in '{s}'; use names from {POOL}, joined with _")
    return [n for n in POOL if n in names]          # fixed pipeline order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MedMNIST/octmnist_64.npz")
    ap.add_argument("--levels", default="study/levels.json")
    ap.add_argument("--out", default="study/label_test_stack")
    ap.add_argument("--clean-ckpt", default="study/label_test/clf.pt")
    ap.add_argument("--stacks", nargs="*", default=["all6"])
    ap.add_argument("--all-fives", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lv = json.loads(Path(a.levels).read_text()); p = lv["prob"]

    stacks = list(a.stacks)
    if a.all_fives:
        stacks += ["_".join(n for n in POOL if n != drop) for drop in POOL]

    d = np.load(a.npz)
    Xtr, ytr = d["train_images"], d["train_labels"].reshape(-1)
    Xva, yva = d["val_images"], d["val_labels"].reshape(-1)
    n_cls = int(ytr.max()) + 1
    labels = list(range(n_cls))

    def score(model, fn):
        pr = predict(model, NpzDS(Xva, yva, fn=fn), dev, a.batch_size, a.workers)
        return 100 * balanced_accuracy_score(yva, pr), 100 * recall_score(yva, pr, average=None, labels=labels)

    base = make_model(n_cls).to(dev)
    ck = Path(a.clean_ckpt)
    if ck.exists():
        base.load_state_dict(torch.load(ck, map_location=dev)); print("loaded clean baseline", ck)
    else:
        print("training clean baseline"); train(base, Xtr, ytr, dev, a.epochs, a.batch_size, a.workers)
    clean, clean_rec = score(base, None)
    print(f"baseline clean validation balanced accuracy {clean:.2f}  per-class {np.round(clean_rec, 1)}", flush=True)

    rows = []
    lines = [f"baseline {clean:.2f}; tolerance {a.tolerance} (any one class {3 * a.tolerance}); each augmentation at p={p}"]
    for s in stacks:
        names = parse(s)
        fns = [from_levels(n, lv) for n in names]
        print(f"\n=== stack {s}: {'+'.join(names)}; training on the stack drawn at p={p}, {a.epochs} epochs", flush=True)
        model = make_model(n_cls).to(dev)
        dl = DataLoader(NpzDS(Xtr, ytr, fn=Stack(fns, p), flip=True), batch_size=a.batch_size, shuffle=True,
                        num_workers=a.workers, drop_last=True)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=a.epochs * len(dl))
        for ep in range(a.epochs):
            model.train(); tot = 0.0
            for x, y in dl:
                x, y = x.to(dev), y.to(dev)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
            print(f"  epoch {ep + 1}/{a.epochs} train loss {tot / len(dl):.4f}", flush=True)
        res = {}
        for lab, fn in (("clean", None), ("drawn", Stack(fns, p)), ("forced", Stack(fns, p, force=True))):
            acc, rec = score(model, fn)
            drop = clean - acc
            worst = float((clean_rec - rec).max())
            ok = drop <= a.tolerance and worst <= 3 * a.tolerance
            res[lab] = ok
            rows.append(dict(stack=s, n_augs=len(names), view=lab, bal_acc=round(acc, 2), drop=round(drop, 2),
                             worst_class_drop=round(worst, 2), passes=ok, **{f"recall_c{c}": round(r, 1) for c, r in enumerate(rec)}))
            print(f"{s[:40]:40s} {lab:7s} bal.acc {acc:6.2f}  drop {drop:6.2f}  worst-class {worst:6.2f}  {'ok' if ok else 'FAIL'}", flush=True)
        lines.append(f"{s:50s} drawn: {'PASS' if res['drawn'] else 'FAIL'}   forced (all on every image): {'pass' if res['forced'] else 'fail'}")
        with (out / "label_test_stack.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
