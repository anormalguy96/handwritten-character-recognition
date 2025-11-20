from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_num_classes(split: str = "mnist") -> int:
    mapping = {
        "mnist": 10,
    }
    if split not in mapping:
        raise ValueError(f"Unknown split: {split}")
    return mapping[split]


def get_emnist_dataloaders(
        data_dir: str = "data",
        split: str = "mnist",
        batch_size: int = 128,
        val_ratio: float = 0.1,
        num_workers: int = 2,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = _build_transform()

    train_val_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(len(train_val_dataset) * val_ratio)
    train_size = len(train_val_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_val_dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader