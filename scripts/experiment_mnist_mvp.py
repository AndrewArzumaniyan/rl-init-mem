from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memristor_simulation.differentiable import DifferentiableMemristorPhysics
from rl_memristor.data import create_mnist_dataloaders
from rl_memristor.mask import create_fc1_programmable_mask
from rl_memristor.policy import LinearGaussianPolicy, MLPGaussianPolicy
from rl_memristor.rollout import run_programming_rollout
from rl_memristor.supervised import evaluate_task_model, train_task_model_step
from rl_memristor.task_model import HardwareNoisyFeedForward
from rl_memristor.train import accuracy_from_logits, train_policy_step


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def limited_batches(loader, limit: int):
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        yield batch


@torch.no_grad()
def evaluate_policy(
    *,
    model: HardwareNoisyFeedForward,
    policy: LinearGaussianPolicy,
    physics: DifferentiableMemristorPhysics,
    loader,
    initial_conductance: torch.Tensor,
    steps: int,
    max_batches: int,
    device: torch.device,
    programmable_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []
    accuracies: list[float] = []
    mean_conductances: list[float] = []
    for inputs, targets in limited_batches(loader, max_batches):
        inputs = inputs.to(device)
        targets = targets.to(device)
        result = run_programming_rollout(
            model=model,
            policy=policy,
            physics=physics,
            inputs=inputs,
            targets=targets,
            initial_conductance=initial_conductance,
            steps=steps,
            criterion=criterion,
            noise_scale=0.0,
            policy_noise_scale=0.0,
            programmable_mask=programmable_mask,
        )
        losses.append(float(result.loss.item()))
        accuracies.append(accuracy_from_logits(result.logits, targets))
        mean_conductances.append(float(result.final_conductance.mean().item()))
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": sum(accuracies) / len(accuracies),
        "mean_conductance": sum(mean_conductances) / len(mean_conductances),
    }


@torch.no_grad()
def evaluate_task_batches(
    *,
    model: HardwareNoisyFeedForward,
    conductance: torch.Tensor,
    loader,
    max_batches: int,
    device: torch.device,
) -> dict[str, float]:
    losses: list[float] = []
    accuracies: list[float] = []
    for inputs, targets in limited_batches(loader, max_batches):
        metrics = evaluate_task_model(
            model=model,
            fc1_conductance=conductance,
            inputs=inputs.to(device),
            targets=targets.to(device),
        )
        losses.append(metrics["loss"])
        accuracies.append(metrics["accuracy"])
    return {"loss": sum(losses) / len(losses), "accuracy": sum(accuracies) / len(accuracies)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MNIST RL memristor MVP experiment.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden1", type=int, default=32)
    parser.add_argument("--hidden2", type=int, default=16)
    parser.add_argument("--pretrain-batches", type=int, default=80)
    parser.add_argument("--policy-batches", type=int, default=60)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--policy-lr", type=float, default=5e-3)
    parser.add_argument("--policy-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--policy-hidden-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--initial-seed", type=int, default=2026)
    parser.add_argument(
        "--initial-mode",
        choices=("constant", "trained", "noisy-trained", "random"),
        default="noisy-trained",
    )
    parser.add_argument("--initial-conductance", type=float, default=0.5)
    parser.add_argument("--initial-noise", type=float, default=0.15)
    parser.add_argument("--programmable-mode", choices=("full", "row", "block"), default="full")
    parser.add_argument("--programmable-row", type=int, default=0)
    parser.add_argument("--programmable-rows", type=int, default=32)
    parser.add_argument("--programmable-cols", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device()
    print(f"device={device}")

    train_loader, test_loader = create_mnist_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )
    model = HardwareNoisyFeedForward(hidden1=args.hidden1, hidden2=args.hidden2).to(device)
    trained_conductance = torch.full(
        model.fc1_conductance_shape,
        0.5,
        device=device,
        requires_grad=True,
    )
    pretrain_optimizer = torch.optim.Adam([trained_conductance, *model.parameters()], lr=args.pretrain_lr)

    print("stage=pretrain_task_model")
    for batch_index, (inputs, targets) in enumerate(limited_batches(train_loader, args.pretrain_batches), start=1):
        metrics = train_task_model_step(
            model=model,
            fc1_conductance=trained_conductance,
            optimizer=pretrain_optimizer,
            inputs=inputs.to(device),
            targets=targets.to(device),
        )
        if batch_index == 1 or batch_index % max(1, args.pretrain_batches // 4) == 0:
            print(f"pretrain_batch={batch_index} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f}")

    task_eval = evaluate_task_batches(
        model=model,
        conductance=trained_conductance,
        loader=test_loader,
        max_batches=args.eval_batches,
        device=device,
    )
    print(f"task_eval loss={task_eval['loss']:.4f} acc={task_eval['accuracy']:.3f}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    programmable_mask = create_fc1_programmable_mask(
        model.fc1_conductance_shape,
        mode=args.programmable_mode,
        row=args.programmable_row,
        rows=args.programmable_rows,
        cols=args.programmable_cols,
        device=device,
    )
    print(
        f"programmable mode={args.programmable_mode} "
        f"count={int(programmable_mask.sum().item())}/{programmable_mask.numel()}"
    )
    initial_generator = torch.Generator(device=device)
    initial_generator.manual_seed(args.initial_seed)
    if args.initial_mode == "constant":
        initial_conductance = trained_conductance.detach().clone()
        initial_conductance = torch.where(
            programmable_mask,
            torch.full_like(initial_conductance, args.initial_conductance),
            initial_conductance,
        )
    elif args.initial_mode == "trained":
        initial_conductance = trained_conductance.detach().clone()
    elif args.initial_mode == "noisy-trained":
        noise = (
            torch.randn(
                trained_conductance.shape,
                generator=initial_generator,
                device=device,
                dtype=trained_conductance.dtype,
            )
            * args.initial_noise
        )
        initial_conductance = torch.where(
            programmable_mask,
            trained_conductance.detach() + noise,
            trained_conductance.detach(),
        ).clamp(model.g_min, model.g_max)
    else:
        random_conductance = torch.rand(
            model.fc1_conductance_shape,
            generator=initial_generator,
            device=device,
        )
        initial_conductance = torch.where(programmable_mask, random_conductance, trained_conductance.detach())

    if args.policy_type == "linear":
        policy = LinearGaussianPolicy(state_dim=8, action_limit=0.2).to(device)
    else:
        policy = MLPGaussianPolicy(
            state_dim=8,
            hidden_dim=args.policy_hidden_dim,
            num_hidden_layers=args.policy_hidden_layers,
            action_limit=0.2,
        ).to(device)
    with torch.no_grad():
        if isinstance(policy, LinearGaussianPolicy):
            policy.mean.weight.zero_()
            policy.mean.bias.zero_()
        else:
            last_layer = policy.mean[-1]
            last_layer.weight.zero_()
            last_layer.bias.zero_()
    physics = DifferentiableMemristorPhysics().to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)

    initial_task_eval = evaluate_task_batches(
        model=model,
        conductance=initial_conductance,
        loader=test_loader,
        max_batches=args.eval_batches,
        device=device,
    )
    print(
        f"initial_conductance mode={args.initial_mode} "
        f"loss={initial_task_eval['loss']:.4f} acc={initial_task_eval['accuracy']:.3f}"
    )

    before = evaluate_policy(
        model=model,
        policy=policy,
        physics=physics,
        loader=test_loader,
        initial_conductance=initial_conductance,
        steps=args.steps,
        max_batches=args.eval_batches,
        device=device,
    )
    print(
        f"policy_before loss={before['loss']:.4f} acc={before['accuracy']:.3f} "
        f"mean_G={before['mean_conductance']:.4f}"
    )

    print("stage=train_policy")
    for batch_index, (inputs, targets) in enumerate(limited_batches(train_loader, args.policy_batches), start=1):
        metrics = train_policy_step(
            model=model,
            policy=policy,
            physics=physics,
            optimizer=policy_optimizer,
            inputs=inputs.to(device),
            targets=targets.to(device),
            initial_conductance=initial_conductance,
            steps=args.steps,
            noise_scale=0.0,
            policy_noise_scale=0.0,
            programmable_mask=programmable_mask,
        )
        if batch_index == 1 or batch_index % max(1, args.policy_batches // 4) == 0:
            print(
                f"policy_batch={batch_index} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f} "
                f"mean_action={metrics['mean_action']:.4f} mean_G={metrics['mean_conductance']:.4f}"
            )

    after = evaluate_policy(
        model=model,
        policy=policy,
        physics=physics,
        loader=test_loader,
        initial_conductance=initial_conductance,
        steps=args.steps,
        max_batches=args.eval_batches,
        device=device,
        programmable_mask=programmable_mask,
    )
    print(
        f"policy_after loss={after['loss']:.4f} acc={after['accuracy']:.3f} "
        f"mean_G={after['mean_conductance']:.4f}"
    )


if __name__ == "__main__":
    main()
