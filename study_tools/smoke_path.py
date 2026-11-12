#!/usr/bin/env python3
"""Checks after add_pathmnist.py, on the real PathMNIST images (CPU only, about a minute).

    python study_tools/smoke_path.py

1. data: split sizes and class counts (the smallest validation class sets the balanced-subset size)
2. normalisation: per-channel mean and std of the training images against the constants in the code
3. every pool operation at every ladder level: output size, timing, and an image grid -> study/path/gallery.png
4. the full augmentation pipeline built by the training code from a yaml with all six operations on
5. the training and online-classifier datasets as main_pretrain.py builds them, and the scoring scripts' dataset
"""
import os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, os.getcwd())
import path_pool as PP

import numpy as np
from PIL import Image, ImageDraw


def main():
    out = Path("study/path"); out.mkdir(parents=True, exist_ok=True)
    d = np.load("MedMNIST/pathmnist_64.npz")
    print("1. data")
    for s in ("train", "val", "test"):
        y = d[f"{s}_labels"].reshape(-1)
        print(f"   {s:5s} {d[f'{s}_images'].shape}  class counts {np.bincount(y).tolist()}")
    yv = d["val_labels"].reshape(-1)
    print(f"   smallest validation class: {np.bincount(yv).min()} images (balanced subsets use this many per class)")

    print("2. normalisation")
    from MedMNIST.pathmnist_dataset import PATHMNIST_MEAN, PATHMNIST_STD
    X = d["train_images"][np.random.RandomState(0).choice(len(d["train_images"]), min(20000, len(d["train_images"])), replace=False)].astype(np.float64) / 255
    m, s = X.mean((0, 1, 2)), X.std((0, 1, 2))
    print(f"   measured mean {np.round(m, 4).tolist()}  std {np.round(s, 4).tolist()}")
    print(f"   in code      mean {list(PATHMNIST_MEAN)}  std {list(PATHMNIST_STD)}")
    ok = np.abs(m - PATHMNIST_MEAN).max() < 0.01 and np.abs(s - PATHMNIST_STD).max() < 0.01
    print("   OK" if ok else "   MISMATCH: run python study_tools/fix_path_norm.py before training")

    print("3. pool operations")
    rng = np.random.RandomState(1)
    picks = [int(rng.choice(np.where(d["val_labels"].reshape(-1) == c)[0])) for c in (0, 2, 3, 5, 8)]
    imgs = [Image.fromarray(d["val_images"][i]) for i in picks]
    rows = [("original", [im for im in imgs])]
    for name in PP.POOL:
        for lab, v, fn in PP.ladder(name):
            t = time.time(); outs = [fn(im) for im in imgs for _ in range(20)][::20]; dt = (time.time() - t) / (20 * len(imgs)) * 1000
            assert all(o.size == (64, 64) and o.mode == "RGB" for o in outs), (name, lab)
            rows.append((f"{name}: {lab}", outs))
            print(f"   {name:9s} {lab:16s} {dt:5.2f} ms/image")
    sc, pad, lw = 2, 4, 150
    W = lw + len(imgs) * (64 * sc + pad)
    H = len(rows) * (64 * sc + pad)
    grid = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(grid)
    for r, (lab, ims) in enumerate(rows):
        y0 = r * (64 * sc + pad)
        dr.text((4, y0 + 56), lab, fill="black")
        for c, im in enumerate(ims):
            grid.paste(im.resize((64 * sc, 64 * sc), Image.NEAREST), (lw + c * (64 * sc + pad), y0))
    grid.save(out / "gallery.png"); print(f"   wrote {out / 'gallery.png'}")

    print("4. training pipeline with all six operations on")
    from omegaconf import OmegaConf
    from MedMNIST.pretrain_dataloader import build_transform_pipeline, prepare_datasets, NCropAugmentation
    mid = {"contrast": 0.8, "rotflip": 1, "stain": 0.05, "defocus": 1.5, "pixelate": 0.8, "jpeg": 70}
    sys.path.insert(0, str(HERE))
    from make_path_runs import render_aug
    cfg = OmegaConf.create(render_aug(set(PP.POOL), mid))[0]
    T = build_transform_pipeline("pathmnist", cfg)
    print("   " + str(T).replace("\n", "\n   "))
    t = time.time(); v = [T(im) for im in imgs for _ in range(100)]; dt = (time.time() - t) / len(v) * 1000
    print(f"   {dt:.2f} ms per view; output {tuple(v[0].shape)}; finite {all(bool(x.isfinite().all()) for x in v)}")

    print("5. datasets as the code builds them")
    ds = prepare_datasets("pathmnist", NCropAugmentation(T, 2), train_data_path="./MedMNIST")
    idx, views, lab = ds[0]
    print(f"   pretraining set: {len(ds)} images; item = (index, {len(views)} views of {tuple(views[0].shape)}, label {lab})")
    from MedMNIST.classification_dataloader import prepare_data
    _, val_loader = prepare_data("pathmnist", train_data_path="./MedMNIST", val_data_path="./MedMNIST", batch_size=64, num_workers=0)
    print(f"   online-classifier validation set: {len(val_loader.dataset)} images")
    import eval_run as ER
    ER.set_dataset("pathmnist")
    print(f"   scoring scripts: {ER.DATASET}, {ER.N_CLASSES} classes, {len(ER.DS(split='val'))} validation images, "
          f"results in {ER.STUDY_DIR}/")
    print("all checks done")


if __name__ == "__main__":
    main()
