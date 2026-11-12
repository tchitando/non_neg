#!/usr/bin/env python3
"""Add PathMNIST to the study pipeline (run once, from the repository root).

    python study_tools/add_pathmnist.py            # check, patch, download the data
    python study_tools/add_pathmnist.py --check    # only check that the patch applies; change nothing

What it changes (each file is backed up as <file>.bak_before_path first):
    MedMNIST/pretrain_dataloader.py       PathMNIST normalisation and dataset; the three new pool operations switch on
    MedMNIST/classification_dataloader.py PathMNIST for the online classifier's validation loader
    MedMNIST/args_pretrain.py             'pathmnist' as a supported dataset with 9 classes
    MedMNIST/medmnistc_transforms.py      RotFlip, StainJitterHED, DefocusRadius and their yaml keys
It also writes MedMNIST/pathmnist_dataset.py (shipped next to this script) and downloads MedMNIST/pathmnist_64.npz.
OCT runs are unaffected: OCT yamls never switch the new operations on, and all OCT code paths are unchanged.
Every edit is an exact text replacement; if any anchor is not found exactly once, nothing is written.
"""
import argparse, json, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARKERS = {"medmnistc_transforms.py": "class StainJitterHED"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path("MedMNIST")
    if not root.is_dir():
        sys.exit("run this from the repository root (the folder that contains MedMNIST/)")
    hunks = json.loads((HERE / "add_pathmnist_hunks.json").read_text())
    plan = {}
    for name, hs in hunks.items():
        p = root / name
        s = p.read_text()
        if MARKERS.get(name, "PathMNIST study") in s:
            print(f"{p}: already patched, skipped"); continue
        for old, new in hs:
            n = s.count(old)
            if n != 1:
                sys.exit(f"{p}: an anchor was found {n} times (expected 1); nothing was changed.\nAnchor starts:\n{old[:300]}")
            s = s.replace(old, new)
        plan[p] = s
        print(f"{p}: {len(hs)} edits apply cleanly")
    if a.check:
        print("check only: nothing written"); return
    for p, s in plan.items():
        shutil.copy2(p, p.with_name(p.name + ".bak_before_path"))
        p.write_text(s)
        print(f"patched {p} (backup {p.name}.bak_before_path)")
    src, dst = HERE / "pathmnist_dataset.py", root / "pathmnist_dataset.py"
    if not dst.exists() or dst.read_text() != src.read_text():
        if dst.exists():
            shutil.copy2(dst, dst.with_name(dst.name + ".bak_before_path"))
        shutil.copy2(src, dst); print(f"wrote {dst}")
    sys.path.insert(0, ".")
    from MedMNIST.pathmnist_dataset import ensure_npz
    path = ensure_npz("MedMNIST")
    print(f"data: {path} ({path.stat().st_size / 1e6:.0f} MB)")
    print("done. Next: python study_tools/smoke_path.py")


if __name__ == "__main__":
    main()
