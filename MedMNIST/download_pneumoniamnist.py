"""
Download PneumoniaMNIST at 64x64 resolution and save as pneumoniamnist_64.npz.

Usage:
    pip install medmnist
    python MedMNIST/download_pneumoniamnist.py
"""
import numpy as np
from pathlib import Path

try:
    from medmnist import PneumoniaMNIST
except ImportError:
    raise SystemExit("Install medmnist first:  pip install medmnist")

ROOT = Path(__file__).parent
SIZE = 64

print(f"Downloading PneumoniaMNIST ({SIZE}x{SIZE}) to {ROOT} ...")

splits = {}
for split in ("train", "val", "test"):
    ds = PneumoniaMNIST(split=split, download=True, size=SIZE, root=str(ROOT))
    images = np.stack([np.array(img) for img, _ in ds])   # (N, H, W) uint8
    labels = np.array([int(label) for _, label in ds])     # (N,)
    splits[f"{split}_images"] = images
    splits[f"{split}_labels"] = labels[:, None]            # (N, 1) to match MedMNIST convention
    print(f"  {split}: {len(ds)} samples, images shape: {images.shape}")

out = ROOT / "pneumoniamnist_64.npz"
np.savez(out, **splits)
print(f"\nSaved to {out}")

# ── Compute normalization statistics ──────────────────────────────────────────
train_imgs = splits["train_images"].astype(np.float32) / 255.0
mean = train_imgs.mean()
std  = train_imgs.std()
print(f"\nNormalization stats (grayscale, replicate to all channels):")
print(f"  PNEUMONIAMNIST_MEAN = ({mean:.4f}, {mean:.4f}, {mean:.4f})")
print(f"  PNEUMONIAMNIST_STD  = ({std:.4f}, {std:.4f}, {std:.4f})")
print("\nUpdate MedMNIST/pneumoniamnist_dataset.py if these differ from the defaults.")
