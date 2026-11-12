#!/usr/bin/env python3
"""Label-preservation test: does a level of an augmentation still leave the diagnosis readable?

A small supervised ResNet-18 is trained once on CLEAN OCT training images. It is then scored on the
validation images after each augmentation at each level. The drop in balanced accuracy against the clean
validation score says how much of the label information a level destroys. Nothing here touches the test split.

    python label_test.py --npz MedMNIST/octmnist_64.npz --out study/label_test
    python label_test.py --npz MedMNIST/octmnist_64.npz --out study/label_test --combo speckle contrast blur motion pixelate jpeg

Outputs <out>/label_test.csv (balanced accuracy, drop, per-class recall, pass/fail per level),
<out>/summary.txt (strongest level that passes for each augmentation) and <out>/clf.pt (cached classifier).
A level passes when the drop is at most --tolerance points (default 2.0). Per-class recall is kept so a
level that wipes out one class but hides in the average is visible. Always read it next to the image grid.
"""
import argparse, csv, json, random
from pathlib import Path

# ImageMagick must be loaded before torch (see medmnistc_transforms.py), so import the augmentations first.
from aug_ladders import POOL, ladder, training_blur, from_levels

import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score

MEAN, STD = 0.1894, 0.2071


class NpzDS(Dataset):
    def __init__(self, X, y, fn=None, flip=False):
        self.X, self.y, self.fn, self.flip = X, y, fn, flip

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        im = Image.fromarray(self.X[i]).convert("RGB")
        if self.fn is not None:
            im = self.fn(im)
        a = (np.asarray(im.convert("L"), dtype=np.float32) / 255.0 - MEAN) / STD
        if self.flip and random.random() < 0.5:
            a = a[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(a))[None].repeat(3, 1, 1), int(self.y[i])


def seed_worker(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


def make_model(n_cls):
    m = models.resnet18(weights=None, num_classes=n_cls)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)   # 64 px input: small stem
    m.maxpool = nn.Identity()
    return m


def predict(model, ds, dev, bs, workers):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, worker_init_fn=seed_worker)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            out.append(model(x.to(dev)).argmax(1).cpu())
    return torch.cat(out).numpy()


def train(model, Xtr, ytr, dev, epochs, bs, workers):
    dl = DataLoader(NpzDS(Xtr, ytr, flip=True), batch_size=bs, shuffle=True, num_workers=workers,
                    worker_init_fn=seed_worker, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * len(dl))
    for ep in range(epochs):
        model.train(); tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"  epoch {ep + 1}/{epochs} train loss {tot / len(dl):.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MedMNIST/octmnist_64.npz")
    ap.add_argument("--out", default="study/label_test")
    ap.add_argument("--levels", default="study/levels.json")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=2.0, help="largest allowed drop in balanced accuracy, in points")
    ap.add_argument("--max-val", type=int, default=0, help="use only this many validation images (0 = all)")
    ap.add_argument("--combo", nargs="*", default=None, help="also test this combination, e.g. --combo speckle contrast")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = np.load(a.npz)
    Xtr, ytr = d["train_images"], d["train_labels"].reshape(-1)
    Xva, yva = d["val_images"], d["val_labels"].reshape(-1)
    if a.max_val:
        sel = np.random.RandomState(a.seed).choice(len(Xva), a.max_val, replace=False)
        Xva, yva = Xva[sel], yva[sel]
    n_cls = int(ytr.max()) + 1

    model = make_model(n_cls).to(dev)
    ck = out / "clf.pt"
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=dev)); print("loaded", ck)
    else:
        print(f"training clean classifier on {len(Xtr)} images for {a.epochs} epochs on {dev}")
        train(model, Xtr, ytr, dev, a.epochs, a.batch_size, a.workers)
        torch.save(model.state_dict(), ck)

    def score(fn):
        p = predict(model, NpzDS(Xva, yva, fn=fn), dev, a.batch_size, a.workers)
        return 100 * balanced_accuracy_score(yva, p), 100 * recall_score(yva, p, average=None, labels=list(range(n_cls)))

    clean, clean_rec = score(None)
    print(f"clean validation balanced accuracy {clean:.2f}  per-class {np.round(clean_rec, 1)}")
    rows = [dict(aug="clean", level="-", bal_acc=round(clean, 2), drop=0.0, passes="",
                 **{f"recall_c{c}": round(r, 1) for c, r in enumerate(clean_rec)})]

    def add(name, label, fn):
        acc, rec = score(fn)
        drop = clean - acc
        # a level fails if the average drop exceeds the tolerance or any single class loses more than 3x the tolerance
        worst = float((clean_rec - rec).max())
        ok = drop <= a.tolerance and worst <= 3 * a.tolerance
        rows.append(dict(aug=name, level=label, bal_acc=round(acc, 2), drop=round(drop, 2), passes=ok,
                         **{f"recall_c{c}": round(r, 1) for c, r in enumerate(rec)}))
        print(f"{name:9s} {label:32s} bal.acc {acc:6.2f}  drop {drop:6.2f}  worst-class drop {worst:6.2f}  {'ok' if ok else 'FAIL'}", flush=True)

    for name in POOL:
        cols = ladder(name) + ([training_blur()] if name == "blur" else [])
        for lab, fn in cols:
            add(name, lab, fn)

    if a.combo is not None:
        lv = json.loads(Path(a.levels).read_text())
        names = a.combo or POOL
        fns = [from_levels(n, lv) for n in names]
        def forced(im):
            for f in fns:
                im = f(im)
            return im
        def drawn(im):
            for f in fns:
                if random.random() < lv["prob"]:
                    im = f(im)
            return im
        add("combo", "+".join(names) + " all forced on", forced)
        add("combo", "+".join(names) + f" each at p={lv['prob']}", drawn)

    with (out / "label_test.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    lines = [f"clean balanced accuracy {clean:.2f}; tolerance {a.tolerance} points (any one class: {3 * a.tolerance})"]
    for name in POOL:
        r = [x for x in rows if x["aug"] == name]
        ok = [x for x in r if x["passes"] is True]
        lines.append(f"{name:9s} strongest passing level: {ok[-1]['level'] if ok else 'NONE (every level fails)'}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
