from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .evaluate import relative_percentage_difference


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3


@dataclass(slots=True)
class TrainResult:
    best_epoch: int
    best_validation_rpd: float
    model_state_dict: dict[str, Tensor]


def evaluate_validation_rpd(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    outputs: list[Tensor] = []
    targets: list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for inputs, target in dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            outputs.append(model(inputs).squeeze(-1).cpu())
            targets.append(target.cpu())
    predictions = torch.cat(outputs)
    reference = torch.cat(targets)
    return float(relative_percentage_difference(predictions, reference).item())


def train_predictor(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    config: TrainConfig,
    device: torch.device | None = None,
) -> TrainResult:
    runtime_device = device or torch.device("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    best_epoch = 0
    best_validation_rpd = float("inf")
    best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.to(runtime_device)
    for epoch in range(config.epochs):
        model.train()
        for inputs, target in train_dataloader:
            inputs = inputs.to(runtime_device)
            target = target.to(runtime_device)
            optimizer.zero_grad()
            prediction = model(inputs).squeeze(-1)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

        validation_rpd = evaluate_validation_rpd(model, validation_dataloader, runtime_device)
        if validation_rpd < best_validation_rpd:
            best_epoch = epoch
            best_validation_rpd = validation_rpd
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    return TrainResult(
        best_epoch=best_epoch,
        best_validation_rpd=best_validation_rpd,
        model_state_dict=best_state_dict,
    )
