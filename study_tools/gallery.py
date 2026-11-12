#!/usr/bin/env python3
"""Contact sheets and numbers for looking at every augmentation before any training.

    python gallery.py --npz MedMNIST/octmnist_64.npz --out study/gallery
    python gallery.py --synthetic --out study/gallery_test      # no data needed, for a quick check

Writes, for each of the six augmentations, <out>/<name>.png : rows are images (3 per class, 4 classes),
columns are the original and levels 1..5. Also <out>/combo.png (the whole stack, as it would appear
in training and the worst case with all six forced on) and <out>/gallery_stats.csv
(SSIM to the original, mean-intensity shift, contrast ratio per level).
Open the PNGs and judge them with the annotators / a clinician: does a level still look like an OCT scan?
"""
import argparse, csv, json, random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from aug_ladders import POOL, ladder, training_blur, from_levels

CLASSES = ["CNV", "DME", "drusen", "normal"]


def load(npz, split, n_per_class, seed):
    d = np.load(npz)
    X, y = d[f"{split}_images"], d[f"{split}_labels"].reshape(-1)
    rng = np.random.RandomState(seed)
    idx, labs = [], []
    for c in range(4):
        sel = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx += list(sel); labs += [c] * n_per_class
    return [Image.fromarray(X[i]).convert("RGB") for i in idx], labs


def synthetic(n_per_class, seed):
    """Layered, speckled stripes that vaguely resemble a B-scan. Only for testing the script."""
    rng = np.random.RandomState(seed)
    imgs, labs = [], []
    for c in range(4):
        for _ in range(n_per_class):
            a = rng.rand(64, 64) * 40 + 20
            for k in range(3):
                r = int(15 + 12 * k + rng.randint(-2, 3))
                a[r:r + 4 + c, :] += 120 - 20 * k
            imgs.append(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")); labs.append(c)
    return imgs, labs


def stem_pil(img, rng):
    """PIL version of the fixed stem (random resized crop 0.08-1, flip, brightness) for the combo view only."""
    w, h = img.size
    s = rng.uniform(0.08, 1.0); ar = np.exp(rng.uniform(np.log(3 / 4), np.log(4 / 3)))
    cw, ch = int(min(w, round((s * w * h * ar) ** 0.5))), int(min(h, round((s * w * h / ar) ** 0.5)))
    x0, y0 = rng.randint(0, w - cw + 1), rng.randint(0, h - ch + 1)
    out = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.BICUBIC)
    if rng.rand() < 0.5:
        out = out.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.rand() < 0.8:
        out = Image.fromarray(np.clip(np.asarray(out) * rng.uniform(0.2, 1.8), 0, 255).astype(np.uint8))
    return out


def gray(im):
    return np.asarray(im.convert("L")).astype(np.float32)


def stats(orig, aug):
    a, b = gray(orig), gray(aug)
    return dict(ssim=float(ssim(a, b, data_range=255)), mean_shift=float(b.mean() - a.mean()),
                contrast_ratio=float(b.std() / max(a.std(), 1e-6)))


def sheet(imgs, labs, cols, path, title):
    n = len(imgs)
    fig, axes = plt.subplots(n, len(cols) + 1, figsize=(1.35 * (len(cols) + 1), 1.35 * n + 0.6))
    for r in range(n):
        axes[r, 0].imshow(imgs[r], cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(CLASSES[labs[r]], rotation=0, ha="right", va="center", fontsize=7)
        for c, (_, fn) in enumerate(cols):
            axes[r, c + 1].imshow(fn(imgs[r]), cmap="gray", vmin=0, vmax=255)
    for c, t in enumerate(["original"] + [l for l, _ in cols]):
        axes[0, c].set_title(t, fontsize=6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MedMNIST/octmnist_64.npz")
    ap.add_argument("--split", default="val")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--levels", default="study/levels.json", help="for combo.png; defaults are used if missing")
    ap.add_argument("--out", default="study/gallery")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    imgs, labs = synthetic(a.per_class, a.seed) if a.synthetic else load(a.npz, a.split, a.per_class, a.seed)

    rows = []
    for name in POOL:
        cols = ladder(name)
        if name == "blur":
            cols = cols + [training_blur()]
        sheet(imgs, labs, cols, out / f"{name}.png", f"{name}: original and levels 1-5")
        for lab, fn in cols:
            ss = [stats(im, fn(im)) for im in imgs]
            rows.append(dict(aug=name, level=lab, **{k: round(float(np.mean([s[k] for s in ss])), 4) for k in ss[0]}))
    with (out / "gallery_stats.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # whole stack, as in training (each extra aug at p) and the worst case (all six forced on)
    lv_path = Path(a.levels)
    from make_runs import DEFAULT_LEVELS
    lv = json.loads(lv_path.read_text()) if lv_path.exists() else DEFAULT_LEVELS
    rng = np.random.RandomState(a.seed)
    fns = [from_levels(n, lv) for n in POOL]
    def trained(im):
        im = stem_pil(im, rng)
        for fn in fns:
            if rng.rand() < lv["prob"]:
                im = fn(im)
        return im
    def worst(im):
        im = stem_pil(im, rng)
        for fn in fns:
            im = fn(im)
        return im
    sheet(imgs, labs, [(f"train draw {i + 1}", trained) for i in range(3)] + [(f"all six on {i + 1}", worst) for i in range(3)],
          out / "combo.png", "full stack: random draws as in training (p=%.1f) and all six forced on" % lv["prob"])
    print(f"wrote {len(POOL)} sheets, combo.png and gallery_stats.csv to {out}")


if __name__ == "__main__":
    main()
