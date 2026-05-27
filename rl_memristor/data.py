from __future__ import annotations

import torch
from torch import Tensor


def create_synthetic_mnist_batch(
    *,
    batch_size: int,
    device: torch.device | str = "cpu",
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    inputs = torch.randn(batch_size, 1, 28, 28, generator=generator).to(device)
    targets = torch.randint(0, 10, (batch_size,), generator=generator).to(device)
    return inputs, targets


def create_mnist_dataloaders(
    *,
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for MNIST dataloaders. Install it or use "
            "create_synthetic_mnist_batch for dependency-free smoke tests."
        ) from exc

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader
