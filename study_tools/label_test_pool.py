#!/usr/bin/env python3
"""Label-preservation test for the PathMNIST pool (same design and pass rule as the OCT test, label_test_aug.py).

For each operation a ResNet-18 is trained on training images that carry the operation (a random level, p 0.5),
then scored on validation images with the operation forced on at each level. A level passes when the drop in
balanced accuracy against a clean classifier on clean images is at most --tolerance points and no class loses more
than 3 x --tolerance points of recall. The validation split only; the test split is never read.

    CUDA_VISIBLE_DEVICES=1 python study_tools/label_test_pool.py                      # all six operations
    CUDA_VISIBLE_DEVICES=1 python study_tools/label_test_pool.py --augs stain defocus # a subset

Writes <out>/label_test.csv, <out>/summary.txt and <out>/levels.json (the levels chosen by the rule in
path_pool.py). Results are written after each operation, so a partial run is usable.
"""
import argparse, csv, json, os, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, os.getcwd())
import path_pool as PP                     # imports the transforms (and ImageMagick) before torch

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score

from MedMNIST.pathmnist_dataset import PATHMNIST_MEAN, PATHMNIST_STD

MEAN = np.array(PATHMNIST_MEAN, dtype=np.float32)
STD = np.array(PATHMNIST_STD, dtype=np.float32)


class NpzRGB(Dataset):
    def __init__(self, X, y, fn=None, flip=False):
        self.X, self.y, self.fn, self.flip = X, y, fn, flip

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        im = Image.fromarray(self.X[i]).convert("RGB")
        if self.fn is not None:
            im = self.fn(im)
        a = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD           # H x W x 3
        if self.flip and random.random() < 0.5:
            a = a[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))), int(self.y[i])


class RandomLevel:
    def __init__(self, fns, p=0.5):
        self.fns, self.p = fns, p

    def __call__(self, img):
        return random.choice(self.fns)(img) if random.random() < self.p else img


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def make_model(n_cls):
    m = models.resnet18(weights=None, num_classes=n_cls)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)        # 64 px input: small stem, as in the OCT test
    m.maxpool = nn.Identity()
    return m


def train(model, ds, dev, epochs, bs, workers):
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers, worker_init_fn=seed_worker, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(dl))
    for ep in range(epochs):
        model.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"  epoch {ep + 1}/{epochs} train loss {tot / len(dl):.4f}", flush=True)


def predict(model, ds, dev, bs, workers):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, worker_init_fn=seed_worker)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            out.append(model(x.to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MedMNIST/pathmnist_64.npz")
    ap.add_argument("--out", default="study/path/label_test")
    ap.add_argument("--augs", nargs="*", default=PP.POOL)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.npz)
    Xtr, ytr = d["train_images"], d["train_labels"].reshape(-1)
    Xva, yva = d["val_images"], d["val_labels"].reshape(-1)
    n_cls = int(ytr.max()) + 1
    labels = list(range(n_cls))
    print(f"PathMNIST: {len(Xtr)} train, {len(Xva)} validation images, {n_cls} classes, on {dev}", flush=True)

    def score(model, fn):
        p = predict(model, NpzRGB(Xva, yva, fn=fn), dev, a.batch_size, a.workers)
        return 100 * balanced_accuracy_score(yva, p), 100 * recall_score(yva, p, average=None, labels=labels)

    base = make_model(n_cls).to(dev)
    ck = out / "clf_clean.pt"
    if ck.exists():
        base.load_state_dict(torch.load(ck, map_location=dev)); print("loaded clean baseline", ck)
    else:
        print("training the clean baseline", flush=True)
        train(base, NpzRGB(Xtr, ytr, flip=True), dev, a.epochs, a.batch_size, a.workers)
        torch.save(base.state_dict(), ck)
    clean, clean_rec = score(base, None)
    print(f"baseline: clean validation balanced accuracy {clean:.2f}, per class {np.round(clean_rec, 1)}", flush=True)

    csv_path, levels_path = out / "label_test.csv", out / "levels.json"
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    rows = [r for r in rows if r["aug"] not in a.augs]
    chosen = json.loads(levels_path.read_text()) if levels_path.exists() else {}
    for name in a.augs:
        lad = PP.ladder(name)
        print(f"\n=== {name}: training on images with a random level (p 0.5), {a.epochs} epochs", flush=True)
        model = make_model(n_cls).to(dev)
        train(model, NpzRGB(Xtr, ytr, fn=RandomLevel([fn for _, _, fn in lad], 0.5), flip=True), dev, a.epochs,
              a.batch_size, a.workers)
        passing = []
        for lab, v, fn in [("clean", None, None)] + lad:
            acc, rec = score(model, fn)
            drop, worst = clean - acc, float((clean_rec - rec).max())
            ok = drop <= a.tolerance and worst <= 3 * a.tolerance
            if ok and v is not None:
                passing.append(v)
            rows.append(dict(aug=name, level=lab, value="" if v is None else v, bal_acc=round(acc, 2), drop=round(drop, 2),
                             worst_class_drop=round(worst, 2), passes=ok,
                             **{f"recall_c{c}": round(r, 1) for c, r in enumerate(rec)}))
            print(f"{name:9s} {lab:16s} bal.acc {acc:6.2f}  drop {drop:6.2f}  worst class {worst:6.2f}  {'ok' if ok else 'FAIL'}",
                  flush=True)
        chosen[name] = PP.choose_level(name, passing)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        levels_path.write_text(json.dumps(chosen, indent=2))
        lines = [f"clean baseline {clean:.2f}; tolerance {a.tolerance} points (any one class {3 * a.tolerance})"]
        for n in PP.POOL:
            if n in chosen:
                passed = [r["level"] for r in rows if r["aug"] == n and str(r["passes"]) == "True" and r["level"] != "clean"]
                lines.append(f"{n:9s} passing: {', '.join(passed) if passed else 'NONE'}   ->  chosen: {chosen[n]}")
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + (out / "summary.txt").read_text())


if __name__ == "__main__":
    main()
