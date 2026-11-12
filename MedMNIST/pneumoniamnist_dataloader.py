"""
Dataloaders for PneumoniaMNIST pretraining and evaluation.

Usage:
    from MedMNIST.pneumoniamnist_dataloader import make_pretrain_loader, make_eval_loaders
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pneumoniamnist_dataset import PneumoniaMnistDataset, PNEUMONIAMNIST_MEAN, PNEUMONIAMNIST_STD

IMAGE_SIZE = 64


def pretrain_transform():
    """Two-crop augmentation pipeline for SSL pretraining."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)   # no hue: image is grayscale
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=PNEUMONIAMNIST_MEAN, std=PNEUMONIAMNIST_STD),
    ])


def eval_transform():
    """Clean transform for linear eval / KNN evaluation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=PNEUMONIAMNIST_MEAN, std=PNEUMONIAMNIST_STD),
    ])


def make_pretrain_loader(batch_size: int = 256, num_workers: int = 4, root=None):
    """Returns a DataLoader over the training split with SSL augmentations."""
    t = pretrain_transform()
    dataset = PneumoniaMnistDataset(split="train", transform=t, root=root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def make_eval_loaders(batch_size: int = 256, num_workers: int = 4, root=None):
    """Returns (train_loader, val_loader, test_loader) with clean transforms."""
    t = eval_transform()
    loaders = {}
    for split in ("train", "val", "test"):
        ds = PneumoniaMnistDataset(split=split, transform=t, root=root)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders["train"], loaders["val"], loaders["test"]
