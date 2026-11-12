import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

OCTMNIST_MEAN = (0.1894, 0.1894, 0.1894)
OCTMNIST_STD  = (0.2071, 0.2071, 0.2071)
DATA_PATH = Path(__file__).parent / "octmnist_64.npz"


class OCTMnistDataset(Dataset):
    """OCT-MNIST loaded from octmnist_64.npz.

    Images are grayscale (H x W uint8), converted to PIL RGB
    (channel triplication) to be compatible with 3-channel pipelines.

    Args:
        split: 'train', 'val', or 'test'
        transform: torchvision transform applied to each PIL image
        root: path to directory containing octmnist_64.npz (defaults to this file's directory)
    """

    def __init__(self, split: str = "train", transform=None, root=None):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        path = Path(root) / "octmnist_64.npz" if root else DATA_PATH
        data = np.load(path)
        self.images = data[f"{split}_images"]                         # (N, 64, 64) uint8
        self.labels = data[f"{split}_labels"].squeeze(1).astype(int)  # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index]).convert("RGB")
        label = int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
