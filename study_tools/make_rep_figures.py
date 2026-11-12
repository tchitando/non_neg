#!/usr/bin/env python3
"""Pictures of the representations, for crop only against brightness 0.8, NCL against SimCLR (OCT, ResNet-50, seed 42).

Run from the repository root on the server (needs the checkpoints and a GPU; a few minutes):

    CUDA_VISIBLE_DEVICES=0 python study_tools/make_rep_figures.py
    CUDA_VISIBLE_DEVICES=0 python study_tools/make_rep_figures.py --split val --per-class 250 --out study/summary/figs_rep
    python study_tools/make_rep_figures.py --selftest          # random data, no GPU or checkpoints: checks the plotting

Figures (PNG and PDF in --out):
    rep1_unit_heatmaps     projector units x classes: each unit's mean activation per class, rescaled per unit to 0-1,
                           units sorted by preferred class; NCL's dead units (never active) are counted in the title
    rep2_top_images_<policy>   for each class, the projector unit whose 9 most activating images are most often that
                           class (same rule for both methods); 3 x 3 mosaic, border = the image's true class
    rep3_umap              UMAP of the backbone features h (class-balanced subset), coloured by class (top row) and by
                           each image's mean brightness (bottom row); falls back to t-SNE if umap-learn is missing
Numbers (CSV in --out):
    rep_numbers.csv        per run: dead units, class information and brightness information in h
                           (5-fold logistic accuracy for class, 5-fold ridge R^2 for image mean brightness)

The projector output z is read out as the loss sees it (eval_spec.readout: SimCLR l2-normalised; NCL normalise -> ReLU
-> normalise), the same readout as SC, MS and the probes. Only the --split images are used (validation by default).
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
CLASSES = ["CNV", "DME", "drusen", "normal"]
CCOL = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF9900"]
RUNS = [("croponly", "Crop only"), ("croponly_bright", "Brightness 0.8")]
OBJS = [("ncl", "NCL"), ("simclr", "SimCLR")]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Garamond", "EB Garamond", "Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# ---------------------------------------------------------------- data
def load_run(run_id, split, ER, ES, bs, workers):
    hits = ER.find_ckpts(run_id, "trained_models")
    if not hits:
        sys.exit(f"no epoch-99 checkpoint for {run_id}")
    bb, pj, _ = ER.load_model(hits[-1])
    h, z_raw, y = ER.extract(bb, pj, split, False, bs, workers)
    z = ES.readout(z_raw, run_id.startswith("ncl-"))
    del bb, pj
    return h.numpy(), z.numpy(), y.numpy()


def synthetic(n=800, seed=0):
    """Random stand-ins with some class structure, for --selftest."""
    rng = np.random.RandomState(seed)
    y = np.repeat(np.arange(4), n // 4)
    imgs = (rng.rand(n, 64, 64) * 255).astype(np.uint8)
    out = {}
    for k, (pol, _) in enumerate(RUNS):
        for obj, _ in OBJS:
            h = rng.randn(n, 64) + np.eye(4)[y] @ rng.randn(4, 64) * (1 + k)
            z = rng.randn(n, 32) + np.eye(4)[y] @ rng.randn(4, 32) * 2
            if obj == "ncl":
                z = np.maximum(z, 0); z[:, :6] = 0
            out[(pol, obj)] = (h, z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9), y)
    return out, imgs


# ---------------------------------------------------------------- figures
def unit_heatmaps(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(9, 10))
    for i, (obj, oname) in enumerate(OBJS):
        for j, (pol, pname) in enumerate(RUNS):
            ax = axes[i, j]
            _, z, y = data[(pol, obj)]
            alive = (z > 0).any(0) if obj == "ncl" else np.ones(z.shape[1], bool)
            M = np.stack([z[y == c].mean(0) for c in range(4)], 1)[alive]          # units x classes
            lo, hi = M.min(1, keepdims=True), M.max(1, keepdims=True)
            R = (M - lo) / np.where(hi > lo, hi - lo, 1)                          # per unit 0-1
            pref = R.argmax(1)
            margin = np.sort(R, 1)[:, -1] - np.sort(R, 1)[:, -2]
            order = np.lexsort((-margin, pref))
            ax.imshow(R[order], aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks(range(4), CLASSES)
            for t, c in zip(ax.get_xticklabels(), CCOL):
                t.set_color(c)
            ax.set_yticks([])
            for c in range(1, 4):
                ax.axvline(c - 0.5, color="white", lw=0.8)
            n_alive, n_dead = int(alive.sum()), int((~alive).sum())
            ax.set_title(f"{oname}, {pname}\n{n_alive} active units" + (f", {n_dead} dead" if obj == "ncl" else ""), fontsize=11)
            if j == 0:
                ax.set_ylabel("projector units (sorted by preferred class)")
    fig.suptitle("Mean activation per class for each projector unit (rescaled per unit)", y=1.0)
    fig.text(0.5, -0.01, "OCT, ResNet-50, seed 42. Readout as in training. A specialised unit shows one bright cell in its row.",
             ha="center", fontsize=9, color="#666666")
    fig.tight_layout()
    save(fig, out, "rep1_unit_heatmaps")


def pick_units(z, y, k=9):
    """For each class: the unit whose k most activating images are most often that class (ties: higher mean activation)."""
    top = np.argsort(-z, 0)[:k]                                                  # k x units
    picks = []
    for c in range(4):
        share = (y[top] == c).mean(0)
        act = np.take_along_axis(z, top, 0).mean(0)
        u = int(np.lexsort((-act, -share))[0])
        picks.append((u, float(share[u]), top[:, u]))
    return picks


def mosaic(ax, imgs, idx, labels, title):
    tiles = []
    for r in range(3):
        row = []
        for c in range(3):
            row.append(imgs[idx[r * 3 + c]])
        tiles.append(np.concatenate(row, 1))
    ax.imshow(np.concatenate(tiles, 0), cmap="gray", vmin=0, vmax=255)
    s = imgs.shape[1]
    for n, i in enumerate(idx[:9]):
        r, c = divmod(n, 3)
        ax.add_patch(Rectangle((c * s - 0.5, r * s - 0.5), s, s, fill=False, lw=2.5, edgecolor=CCOL[labels[i]]))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def top_images(data, imgs, out):
    for pol, pname in RUNS:
        fig, axes = plt.subplots(4, 2, figsize=(7, 14))
        for j, (obj, oname) in enumerate(OBJS):
            _, z, y = data[(pol, obj)]
            for c, (u, share, idx) in enumerate(pick_units(z, y)):
                mosaic(axes[c, j], imgs, idx, y, f"{oname}: unit {u} → {CLASSES[c]}, {share * 100:.0f}% {CLASSES[c]}")
        for c in range(4):
            axes[c, 0].set_ylabel(CLASSES[c], color=CCOL[c], fontsize=12)
        fig.suptitle(f"{pname}: top 9 images of each class's most selective projector unit\n"
                     "border = true class (" + ", ".join(f"{n}" for n in CLASSES) + ")", y=1.0, fontsize=12)
        fig.tight_layout()
        save(fig, out, f"rep2_top_images_{pol}")


def embed(X, seed=0):
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed).fit_transform(X), "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(X), "t-SNE"


def manifolds(data, imgs, sub, out):
    from sklearn.preprocessing import normalize
    bright = imgs.reshape(len(imgs), -1).mean(1)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    method = "UMAP"
    for j, ((pol, pname), (obj, oname)) in enumerate([(r, o) for o in OBJS for r in RUNS]):
        h, _, y = data[(pol, obj)]
        E, method = embed(normalize(h[sub]))
        ax = axes[0, j]
        for c in range(4):
            m = y[sub] == c
            ax.scatter(E[m, 0], E[m, 1], s=4, color=CCOL[c], alpha=0.7, label=CLASSES[c], rasterized=True)
        ax.set_title(f"{oname}, {pname}", fontsize=11)
        ax2 = axes[1, j]
        sc = ax2.scatter(E[:, 0], E[:, 1], s=4, c=bright[sub], cmap="viridis", alpha=0.8, rasterized=True)
        for a in (ax, ax2):
            a.set_xticks([]); a.set_yticks([])
    axes[0, 0].legend(markerscale=4, fontsize=9, loc="best")
    axes[0, 0].set_ylabel("coloured by class"); axes[1, 0].set_ylabel("coloured by image brightness")
    cb = fig.colorbar(sc, ax=axes[1, :].tolist(), fraction=0.015, pad=0.01)
    cb.set_label("mean pixel intensity")
    fig.suptitle(f"{method} of the backbone features h (class-balanced subset, {len(sub)} images)", y=1.0)
    fig.text(0.5, 0.01, "OCT, ResNet-50, seed 42. Each panel is fitted separately: compare the structure within a panel, "
             "not positions or distances between panels.", ha="center", fontsize=9, color="#666666")
    save(fig, out, "rep3_umap")


def numbers(data, imgs, sub, out):
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    bright = imgs.reshape(len(imgs), -1).mean(1)
    rows = []
    for pol, pname in RUNS:
        for obj, oname in OBJS:
            h, z, y = data[(pol, obj)]
            X = h[sub]
            acc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y[sub], cv=5).mean()
            r2 = cross_val_score(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))), X, bright[sub],
                                 cv=5, scoring="r2").mean()
            dead = int((~(z > 0).any(0)).sum()) if obj == "ncl" else 0
            rows.append(dict(policy=pol, method=oname, dead_units=dead, class_acc_5fold=round(100 * acc, 2),
                             brightness_r2_5fold=round(r2, 3), n_images=len(sub)))
            print(f"{oname:7s} {pname:15s} dead {dead:3d}   class from h {100 * acc:5.1f}%   brightness from h R2 {r2:.3f}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rep_numbers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out / "rep_numbers.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--per-class", type=int, default=250, help="class-balanced subset for the manifolds and the numbers")
    ap.add_argument("--seed-suffix", default="", help="'' = seed 42 runs; e.g. '-s43' for seed 43")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="study/summary/figs_rep")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    if a.selftest:
        data, imgs = synthetic()
        y = data[(RUNS[0][0], "ncl")][2]
    else:
        sys.path.insert(0, str(HERE)); sys.path.insert(0, ".")
        import eval_run as ER
        import eval_spec as ES
        ER.set_dataset("octmnist")
        data = {}
        for pol, _ in RUNS:
            for obj, _ in OBJS:
                rid = f"{obj}-resnet50-octmnist-{pol}{a.seed_suffix}"
                print("loading", rid, flush=True)
                data[(pol, obj)] = load_run(rid, a.split, ER, ES, a.bs, a.workers)
        imgs = ER.DS(split=a.split).images
        y = data[(RUNS[0][0], "ncl")][2]
    rng = np.random.RandomState(0)
    per = min(a.per_class, min(int((y == c).sum()) for c in range(4)))
    sub = np.concatenate([rng.choice(np.where(y == c)[0], per, replace=False) for c in range(4)])
    unit_heatmaps(data, out)
    top_images(data, imgs, out)
    manifolds(data, imgs, sub, out)
    numbers(data, imgs, sub, out)


if __name__ == "__main__":
    main()
