"""PathMNIST dataset wrapper using the MedMNIST v2 API."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import medmnist
from medmnist import INFO

INFO_PATH = INFO["pathmnist"]
CLASS_NAMES = list(INFO_PATH["label"].values())
N_CLASSES   = len(CLASS_NAMES)   # 9

# Per-channel mean/std computed from PathMNIST training split (64×64, RGB)
# H&E stained colorectal histology — channels differ significantly
PATHMNIST_MEAN = (0.7406, 0.5331, 0.7059)
PATHMNIST_STD  = (0.1292, 0.1765, 0.1381)


class PathMNISTDataset(Dataset):
    """PathMNIST loaded via MedMNIST API.

    Images are RGB H&E stained histology patches from NCT-CRC-HE-100K.
    9 tissue types: adipose, background, debris, lymphocytes, mucus,
    smooth muscle, normal mucosa, stroma, adenocarcinoma epithelium.

    Args:
        split:     'train', 'val', or 'test'
        transform: torchvision transform applied to each PIL RGB image
        root:      directory to download/cache the dataset
        size:      image resolution — 28 or 64 (MedMNIST v2)
        download:  download if not already present
    """

    def __init__(self, split="train", transform=None, root=None,
                 size=64, download=True):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        kwargs = dict(split=split, download=download, size=size, as_rgb=True)
        if root is not None:
            kwargs["root"] = root
        self._ds = medmnist.PathMNIST(**kwargs)
        self.transform = transform

        self.images = self._ds.imgs                                  # (N, H, W, 3) uint8
        self.labels = self._ds.labels.squeeze(1).astype(int)        # (N,)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, index):
        img_arr = self.images[index]
        img = Image.fromarray(img_arr)   # already RGB — no .convert() needed
        label = int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
