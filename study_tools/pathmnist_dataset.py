"""PathMNIST (64 x 64, RGB) loaded from pathmnist_64.npz, with the same interface as OCTMnistDataset.

The npz is the MedMNIST+ file. If it is missing, `ensure_npz(root)` downloads it with the medmnist package
(study_tools/add_pathmnist.py calls it once). Images are H&E histology patches, 9 tissue classes.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Per-channel mean and std of the 64 px training images (checked by study_tools/smoke_path.py).
PATHMNIST_MEAN = (0.7406, 0.5331, 0.7059)
PATHMNIST_STD = (0.1292, 0.1765, 0.1381)
N_CLASSES = 9
DATA_PATH = Path(__file__).parent / "pathmnist_64.npz"


def ensure_npz(root=None):
    """Return the path of pathmnist_64.npz under root (default: this folder), downloading it if missing."""
    path = (Path(root) if root else DATA_PATH.parent) / "pathmnist_64.npz"
    if not path.exists():
        import medmnist
        medmnist.PathMNIST(split="train", download=True, size=64, root=str(path.parent))
    return path


class PathMnistDataset(Dataset):
    """PathMNIST at 64 px.

    Args:
        split: 'train', 'val' or 'test'
        transform: applied to each PIL RGB image
        root: folder containing pathmnist_64.npz (default: this file's folder)
    """

    def __init__(self, split: str = "train", transform=None, root=None):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        path = Path(root) / "pathmnist_64.npz" if root else DATA_PATH
        data = np.load(path)
        self.images = data[f"{split}_images"]                          # (N, 64, 64, 3) uint8
        self.labels = data[f"{split}_labels"].reshape(-1).astype(int)  # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index]).convert("RGB")
        label = int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
