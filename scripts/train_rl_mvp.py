from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memristor_simulation.differentiable import DifferentiableMemristorPhysics
from rl_memristor.data import create_mnist_dataloaders, create_synthetic_mnist_batch
from rl_memristor.policy import LinearGaussianPolicy
from rl_memristor.task_model import HardwareNoisyFeedForward
from rl_memristor.train import train_policy_step


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL memristor MVP on synthetic MNIST-shaped data.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--hidden1", type=int, default=16)
    parser.add_argument("--hidden2", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dataset", choices=("synthetic", "mnist"), default="synthetic")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device()
    model = HardwareNoisyFeedForward(hidden1=args.hidden1, hidden2=args.hidden2).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    policy = LinearGaussianPolicy(state_dim=8, action_limit=0.2).to(device)
    physics = DifferentiableMemristorPhysics().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    initial_conductance = torch.full(model.fc1_conductance_shape, 0.5, device=device)
    if args.dataset == "mnist":
        train_loader, _ = create_mnist_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            download=True,
        )
        batch_iter = iter(train_loader)
    else:
        inputs, targets = create_synthetic_mnist_batch(
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
        )
        batch_iter = None

    for epoch in range(1, args.epochs + 1):
        if batch_iter is not None:
            try:
                inputs, targets = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                inputs, targets = next(batch_iter)
            inputs = inputs.to(device)
            targets = targets.to(device)

        metrics = train_policy_step(
            model=model,
            policy=policy,
            physics=physics,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            initial_conductance=initial_conductance,
            steps=args.steps,
            noise_scale=args.noise_scale,
        )
        print(
            f"epoch={epoch} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
            f"mean_action={metrics['mean_action']:.4f} mean_G={metrics['mean_conductance']:.4f}"
        )


if __name__ == "__main__":
    main()
