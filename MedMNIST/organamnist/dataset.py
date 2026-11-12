"""OrganAMNIST dataset wrapper using the MedMNIST v2 API."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import medmnist
from medmnist import INFO

INFO_ORGAN = INFO["organamnist"]
CLASS_NAMES = list(INFO_ORGAN["label"].values())
N_CLASSES   = len(CLASS_NAMES)   # 11

# Per-channel mean/std computed from OrganAMNIST training split (28×28, RGB-replicated)
ORGANAMNIST_MEAN = (0.4677, 0.4677, 0.4677)
ORGANAMNIST_STD  = (0.2479, 0.2479, 0.2479)


class OrganAMNISTDataset(Dataset):
    """OrganAMNIST loaded via MedMNIST API.

    Images are greyscale CT organ slices, tripled to 3 channels for
    compatibility with ImageNet-pretrained 3-channel pipelines.

    Args:
        split:     'train', 'val', or 'test'
        transform: torchvision transform applied to each PIL RGB image
        root:      directory to download/cache the dataset (default: ~/.medmnist)
        size:      image resolution — 28 or 64 (MedMNIST v2 supports both)
        download:  download if not already present
    """

    def __init__(self, split="train", transform=None, root=None,
                 size=64, download=True):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        kwargs = dict(split=split, download=download, size=size)
        if root is not None:
            kwargs["root"] = root
        self._ds = medmnist.OrganAMNIST(**kwargs)
        self.transform = transform

        # cache arrays for fast access (used by eval scripts)
        self.images = self._ds.imgs                                      # (N, H, W) uint8
        self.labels = self._ds.labels.squeeze(1).astype(int)            # (N,)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        img_arr = self.images[index]
        img = Image.fromarray(img_arr).convert("RGB")
        label = int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
