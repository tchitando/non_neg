#!/usr/bin/env python3
"""Set the PathMNIST normalisation constants to the exact per-channel mean and std of the training images.

    python study_tools/fix_path_norm.py            # compute, then update every file that holds the constants
    python study_tools/fix_path_norm.py --check    # compute and show what would change; write nothing

The mean and std are computed over every pixel of the 89,996 training images (64 px). The old values
(mean 0.7406, 0.5331, 0.7059; std 0.1292, 0.1765, 0.1381) are replaced in the dataset class, the pretraining and
online-classifier dataloaders, the run generator and any PathMNIST augmentation yamls already written.
Refuses to run once a PathMNIST run has been claimed, because a run must use one normalisation throughout.
Each changed file is backed up as <file>.bak_norm.
"""
import argparse, glob, shutil, sys
from pathlib import Path

import numpy as np

OLD_MEAN, OLD_STD = "0.7406, 0.5331, 0.7059", "0.1292, 0.1765, 0.1381"
FILES = ["MedMNIST/pathmnist_dataset.py", "MedMNIST/pretrain_dataloader.py", "MedMNIST/classification_dataloader.py",
         "study_tools/make_path_runs.py", "study_tools/pathmnist_dataset.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MedMNIST/pathmnist_64.npz")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    started = glob.glob("study/claims/*pathmnist*")
    if started and not a.check:
        sys.exit(f"{len(started)} PathMNIST run(s) already claimed in study/claims/; not changing the normalisation")
    X = np.load(a.npz)["train_images"]
    s, ss, n = np.zeros(3), np.zeros(3), 0
    for i in range(0, len(X), 5000):
        x = X[i:i + 5000].reshape(-1, 3).astype(np.float64) / 255.0
        s += x.sum(0); ss += (x ** 2).sum(0); n += len(x)
    mean = s / n
    std = np.sqrt(ss / n - mean ** 2)
    new_mean = ", ".join(f"{v:.4f}" for v in mean)
    new_std = ", ".join(f"{v:.4f}" for v in std)
    print(f"training images: {len(X)}; pixels per channel: {n}")
    print(f"mean {new_mean}   (was {OLD_MEAN})")
    print(f"std  {new_std}   (was {OLD_STD})")
    files = FILES + sorted(glob.glob("MedMNIST/scripts/pretrain/pathmnist_study/augmentations/*.yaml"))
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        t = p.read_text()
        k = t.count(OLD_MEAN) + t.count(OLD_STD)
        if k == 0:
            print(f"  {f}: nothing to change"); continue
        print(f"  {f}: {k} replacement(s)" + ("" if not a.check else " (check only)"))
        if not a.check:
            shutil.copy2(p, p.with_name(p.name + ".bak_norm"))
            p.write_text(t.replace(OLD_MEAN, new_mean).replace(OLD_STD, new_std))
    if not a.check:
        print("done. Rerun python study_tools/smoke_path.py: section 2 should now say OK")


if __name__ == "__main__":
    main()
