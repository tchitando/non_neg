"""Smoke test for medmnistc_transforms.py: shapes, levels, timing on a synthetic 64x64 grayscale image. Run from the repository root: python study_tools/test_transforms.py"""
import time, sys
import numpy as np
from PIL import Image
import os, sys
sys.path.insert(0, os.getcwd())
try:
    from MedMNIST.medmnistc_transforms import MedMNISTCAugmentation, EXTRA_ORDER
except ImportError:
    from medmnistc_transforms import MedMNISTCAugmentation, EXTRA_ORDER

rng = np.random.RandomState(0)
base = (rng.rand(64, 64) * 80 + 60).astype(np.uint8)
base[20:40, 10:54] = 200
img = Image.fromarray(base).convert("RGB")
ok = True
for name in EXTRA_ORDER:
    for lvl in range(5):
        try:
            aug = MedMNISTCAugmentation(name, lvl)
            t = time.time(); n = 20
            for _ in range(n):
                out = aug(img)
            ms = (time.time() - t) / n * 1000
            d = np.abs(np.asarray(out).astype(float) - np.asarray(img).astype(float)).mean()
            print(f"{name:17s} L{lvl+1} param={aug.param()!s:22s} shape={np.asarray(out).shape} mean|diff|={d:6.2f} {ms:6.2f} ms")
        except Exception as e:
            ok = False
            print(f"{name:17s} L{lvl+1} FAILED: {type(e).__name__}: {e}")
print("ALL OK" if ok else "SOME FAILED")
