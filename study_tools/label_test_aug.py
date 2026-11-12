#!/usr/bin/env python3
"""Label-preservation test, version 2: can the diagnosis still be LEARNED under each augmentation?

label_test.py trains on clean images and then shows the classifier corrupted images. A network that has
never seen noise or blur fails on them even when a person can still read the scan, so that test measures
how brittle the classifier is. This version trains one fresh classifier per augmentation on images that
carry the augmentation (each image gets a random level 1-5 with probability 0.5), then scores it on
validation images with the augmentation forced on at each level. If accuracy at a level stays close to the
clean baseline, the diagnostic information survives that level. This is the question the study needs.

    python study_tools/label_test_aug.py --npz MedMNIST/octmnist_64.npz --out study/label_test_aug
    python study_tools/label_test_aug.py --augs speckle contrast        # only some augmentations

Baseline = the clean-trained classifier from label_test.py (study/label_test/clf.pt), scored on clean
validation images; it is trained if that file is missing. A level passes when the balanced-accuracy drop
from the baseline is at most --tolerance and no class loses more than 3x --tolerance.
Results are written after each augmentation, so a partial run is usable.
"""
import argparse, csv, random
from pathlib import Path

from aug_ladders import POOL, MILD, EXTENDED, ladder, mild_ladder, extended_ladder, training_blur

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score

from label_test import NpzDS, make_model, predict, train


class RandomLevel:
    """With probability p apply one of the augmentation levels, chosen uniformly."""

    def __init__(self, fns, p=0.5):
        self.fns, self.p = fns, p

    def __call__(self, img):
        return random.choice(self.fns)(img) if random.random() < self.p else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MedMNIST/octmnist_64.npz")
    ap.add_argument("--out", default="study/label_test_aug")
    ap.add_argument("--clean-ckpt", default="study/label_test/clf.pt")
    ap.add_argument("--augs", nargs="*", default=POOL)
    ap.add_argument("--mild", action="store_true", help="use the gentler ladders for blur, motion, pixelate, JPEG (and write to study/label_test_aug_mild)")
    ap.add_argument("--extended", action="store_true", help="the 29 Oct ladders: contrast below 0.5, brightness, rotation, "
                    "crop and elastic deformation (writes to study/label_test_aug_ext)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.mild and a.out == "study/label_test_aug":
        a.out = "study/label_test_aug_mild"
        a.augs = [x for x in a.augs if x in MILD] if a.augs == POOL else a.augs
    if a.extended:
        if a.out == "study/label_test_aug":
            a.out = "study/label_test_aug_ext"
        a.augs = EXTENDED if a.augs == POOL else a.augs
        bad = [x for x in a.augs if x not in EXTENDED]
        if bad:
            raise SystemExit(f"--extended takes {EXTENDED}, not {bad}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.npz)
    Xtr, ytr = d["train_images"], d["train_labels"].reshape(-1)
    Xva, yva = d["val_images"], d["val_labels"].reshape(-1)
    n_cls = int(ytr.max()) + 1
    labels = list(range(n_cls))

    def score(model, fn):
        p = predict(model, NpzDS(Xva, yva, fn=fn), dev, a.batch_size, a.workers)
        return 100 * balanced_accuracy_score(yva, p), 100 * recall_score(yva, p, average=None, labels=labels)

    base = make_model(n_cls).to(dev)
    ck = Path(a.clean_ckpt)
    if ck.exists():
        base.load_state_dict(torch.load(ck, map_location=dev)); print("loaded clean baseline", ck)
    else:
        print("training clean baseline"); train(base, Xtr, ytr, dev, a.epochs, a.batch_size, a.workers)
    clean, clean_rec = score(base, None)
    print(f"baseline clean validation balanced accuracy {clean:.2f}  per-class {np.round(clean_rec, 1)}", flush=True)

    rows, lines = [], [f"baseline (clean-trained, clean val) {clean:.2f}; tolerance {a.tolerance} points (any one class {3 * a.tolerance})"]
    for name in a.augs:
        cols = extended_ladder(name) if a.extended else (mild_ladder(name) if (a.mild and name in MILD) else ladder(name))
        print(f"\n=== {name}: training a classifier on images with random levels 1-5 (p=0.5), {a.epochs} epochs", flush=True)
        model = make_model(n_cls).to(dev)
        train_ds_fn = RandomLevel([fn for _, fn in cols], 0.5)
        # train() builds its own dataset without augmentation, so build the loader here
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        dl = DataLoader(NpzDS(Xtr, ytr, fn=train_ds_fn, flip=True), batch_size=a.batch_size, shuffle=True,
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

        passed = []
        extra = [training_blur()] if (name == "blur" and not a.mild) else []
        for lab, fn in [("clean", None)] + cols + extra:
            acc, rec = score(model, fn)
            drop = clean - acc
            worst = float((clean_rec - rec).max())
            ok = drop <= a.tolerance and worst <= 3 * a.tolerance
            if ok and lab in [l for l, _ in cols]:
                passed.append(lab)
            rows.append(dict(aug=name, level=lab, bal_acc=round(acc, 2), drop=round(drop, 2), worst_class_drop=round(worst, 2),
                             passes=ok, **{f"recall_c{c}": round(r, 1) for c, r in enumerate(rec)}))
            print(f"{name:9s} {lab:32s} bal.acc {acc:6.2f}  drop {drop:6.2f}  worst-class {worst:6.2f}  {'ok' if ok else 'FAIL'}", flush=True)
        lines.append(f"{name:9s} passing levels: {', '.join(passed) if passed else 'NONE'}")
        with (out / "label_test_aug.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        (out / "summary.txt").write_text("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
