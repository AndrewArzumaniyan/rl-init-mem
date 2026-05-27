"""Full-network memristor programming experiment.

Sequential strategy: FC3 → FC2 → FC1
Each layer is programmed with a dedicated RL policy while the other layers
use their best available conductances (trained → programmed).

FC1 is programmed block-by-block via divide-and-conquer.
FC2 and FC3 are small enough to program in a single shot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memristor_simulation.differentiable import DifferentiableMemristorPhysics
from rl_memristor.data import create_mnist_dataloaders
from rl_memristor.mask import create_block_mask
from rl_memristor.policy import LinearGaussianPolicy, MLPGaussianPolicy
from rl_memristor.rollout import run_programming_rollout
from rl_memristor.supervised import train_full_hardware_task_model_step
from rl_memristor.task_model import HardwareNoisyFeedForward
from rl_memristor.train import accuracy_from_logits


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def limited_batches(loader, limit: int):
    for i, batch in enumerate(loader):
        if i >= limit:
            break
        yield batch


def make_policy(policy_type: str, hidden_dim: int, num_layers: int, device: torch.device):
    if policy_type == "linear":
        p = LinearGaussianPolicy(state_dim=8, action_limit=0.2)
    else:
        p = MLPGaussianPolicy(state_dim=8, hidden_dim=hidden_dim, num_hidden_layers=num_layers, action_limit=0.2)
    with torch.no_grad():
        if policy_type == "linear":
            p.mean.weight.zero_()
            p.mean.bias.zero_()
        else:
            last = p.mean[-1]
            last.weight.zero_()
            last.bias.zero_()
    return p.to(device)


@torch.no_grad()
def evaluate_full(
    *,
    model: HardwareNoisyFeedForward,
    fc1_cond: Tensor,
    fc2_cond: Tensor,
    fc3_cond: Tensor,
    loader,
    max_batches: int,
    device: torch.device,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []
    model.eval()
    for inputs, targets in limited_batches(loader, max_batches):
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model.forward_with_conductance(inputs, fc1_cond, fc2_cond, fc3_cond)
        loss = criterion(logits, targets)
        losses.append(float(loss.item()))
        accs.append(accuracy_from_logits(logits, targets))
    return {"loss": sum(losses) / len(losses), "accuracy": sum(accs) / len(accs)}


def train_layer_policy(
    *,
    label: str,
    model: HardwareNoisyFeedForward,
    policy,
    physics: DifferentiableMemristorPhysics,
    optimizer: torch.optim.Optimizer,
    train_loader,
    num_batches: int,
    initial_conductance: Tensor,
    active_layer: int,
    fc1_context: Tensor | None,
    fc2_context: Tensor | None,
    fc3_context: Tensor | None,
    steps: int,
    device: torch.device,
    programmable_mask: Tensor | None = None,
    log_every: int = 20,
    deterministic_policy_training: bool = True,
) -> Tensor:
    """Train policy and return conductance from a deterministic rollout on the last batch."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    last_result = None
    last_batch: tuple[Tensor, Tensor] | None = None
    for batch_idx, (inputs, targets) in enumerate(limited_batches(train_loader, num_batches), start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        last_batch = (inputs, targets)
        optimizer.zero_grad()
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
            policy_noise_scale=0.0 if deterministic_policy_training else 1.0,
            programmable_mask=programmable_mask,
            active_layer=active_layer,
            fc1_context=fc1_context,
            fc2_context=fc2_context,
            fc3_context=fc3_context,
        )
        result.loss.backward()
        optimizer.step()
        last_result = result
        if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == num_batches:
            print(
                f"  {label} batch={batch_idx}/{num_batches} "
                f"loss={float(result.loss.item()):.4f} "
                f"acc={accuracy_from_logits(result.logits.detach(), targets):.3f}"
            )
    if last_result is None or last_batch is None:
        return initial_conductance
    inputs, targets = last_batch
    deterministic_result = run_programming_rollout(
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
        active_layer=active_layer,
        fc1_context=fc1_context,
        fc2_context=fc2_context,
        fc3_context=fc3_context,
    )
    return deterministic_result.final_conductance.detach()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-network memristor programming experiment.")
    # Data / model
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    # Pretraining
    parser.add_argument("--pretrain-batches", type=int, default=200)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    # Policy
    parser.add_argument("--policy-type", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--policy-hidden-dim", type=int, default=64)
    parser.add_argument("--policy-hidden-layers", type=int, default=2)
    parser.add_argument("--policy-lr", type=float, default=5e-3)
    parser.add_argument("--initial-noise", type=float, default=0.2)
    parser.add_argument("--initial-seed", type=int, default=2026)
    # FC3 programming (small — 1 shot, full)
    parser.add_argument("--fc3-steps", type=int, default=5)
    parser.add_argument("--fc3-batches", type=int, default=100)
    # FC2 programming (medium)
    parser.add_argument("--fc2-steps", type=int, default=10)
    parser.add_argument("--fc2-batches", type=int, default=100)
    parser.add_argument("--fc2-block-rows", type=int, default=0,
                        help="0 = program FC2 in one shot (full matrix)")
    parser.add_argument("--fc2-block-cols", type=int, default=0)
    # FC1 programming (large — divide and conquer)
    parser.add_argument("--fc1-steps", type=int, default=5)
    parser.add_argument("--fc1-batches-per-block", type=int, default=30)
    parser.add_argument("--fc1-block-rows", type=int, default=32)
    parser.add_argument("--fc1-block-cols", type=int, default=128)
    # Evaluation
    parser.add_argument("--eval-batches", type=int, default=20)
    # Seeds
    parser.add_argument("--seed", type=int, default=42)
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

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------
    model = HardwareNoisyFeedForward(hidden1=args.hidden1, hidden2=args.hidden2).to(device)
    print(
        f"model: FC1={model.fc1_conductance_shape} "
        f"FC2={model.fc2_conductance_shape} "
        f"FC3={model.fc3_conductance_shape}"
    )
    fc1_learnable = torch.full(model.fc1_conductance_shape, 0.5, device=device, requires_grad=True)
    fc2_learnable = torch.full(model.fc2_conductance_shape, 0.5, device=device, requires_grad=True)
    fc3_learnable = torch.full(model.fc3_conductance_shape, 0.5, device=device, requires_grad=True)

    # ------------------------------------------------------------------
    # Pretrain: optimize all hardware conductances plus normalization parameters.
    # ------------------------------------------------------------------
    print(f"\nstage=pretrain  batches={args.pretrain_batches}")
    pretrain_opt = torch.optim.Adam(
        [
            fc1_learnable,
            fc2_learnable,
            fc3_learnable,
            *model.bn1.parameters(),
            *model.bn2.parameters(),
            *model.ln.parameters(),
        ],
        lr=args.pretrain_lr,
    )
    for bi, (inputs, targets) in enumerate(
        limited_batches(train_loader, args.pretrain_batches), start=1
    ):
        metrics = train_full_hardware_task_model_step(
            model=model,
            fc1_conductance=fc1_learnable,
            fc2_conductance=fc2_learnable,
            fc3_conductance=fc3_learnable,
            optimizer=pretrain_opt,
            inputs=inputs.to(device),
            targets=targets.to(device),
        )
        if bi == 1 or bi % max(1, args.pretrain_batches // 4) == 0 or bi == args.pretrain_batches:
            print(f"  pretrain batch={bi} loss={metrics['loss']:.4f} acc={metrics['accuracy']:.3f}")

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Freeze model parameters and detach trained conductances.
    # ------------------------------------------------------------------
    fc1_cond_init = fc1_learnable.detach().clamp(model.g_min, model.g_max)
    fc2_cond_init = fc2_learnable.detach().clamp(model.g_min, model.g_max)
    fc3_cond_init = fc3_learnable.detach().clamp(model.g_min, model.g_max)
    print(
        f"\ntrained conductances: "
        f"FC1 mean={fc1_cond_init.mean():.3f}, "
        f"FC2 mean={fc2_cond_init.mean():.3f}, "
        f"FC3 mean={fc3_cond_init.mean():.3f}"
    )

    physics = DifferentiableMemristorPhysics().to(device)

    # ------------------------------------------------------------------
    # Baseline and noisy evaluation.
    # ------------------------------------------------------------------
    trained_baseline = evaluate_full(
        model=model,
        fc1_cond=fc1_cond_init,
        fc2_cond=fc2_cond_init,
        fc3_cond=fc3_cond_init,
        loader=test_loader,
        max_batches=args.eval_batches,
        device=device,
    )
    print(f"\ntrained_baseline  loss={trained_baseline['loss']:.4f} acc={trained_baseline['accuracy']:.3f}")

    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(args.initial_seed)

    def add_initial_noise(conductance: Tensor) -> Tensor:
        noise = torch.randn(
            conductance.shape,
            generator=noise_generator,
            device=device,
            dtype=conductance.dtype,
        ) * args.initial_noise
        return (conductance + noise).clamp(model.g_min, model.g_max)

    fc1_cond_noisy = add_initial_noise(fc1_cond_init)
    fc2_cond_noisy = add_initial_noise(fc2_cond_init)
    fc3_cond_noisy = add_initial_noise(fc3_cond_init)
    noisy_baseline = evaluate_full(
        model=model,
        fc1_cond=fc1_cond_noisy,
        fc2_cond=fc2_cond_noisy,
        fc3_cond=fc3_cond_noisy,
        loader=test_loader,
        max_batches=args.eval_batches,
        device=device,
    )
    print(f"noisy_baseline    loss={noisy_baseline['loss']:.4f} acc={noisy_baseline['accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Phase 1: Program FC3
    # ------------------------------------------------------------------
    print(f"\nstage=program_FC3  steps={args.fc3_steps} batches={args.fc3_batches}")
    policy_fc3 = make_policy(args.policy_type, args.policy_hidden_dim, args.policy_hidden_layers, device)
    opt_fc3 = torch.optim.Adam(policy_fc3.parameters(), lr=args.policy_lr)

    fc3_cond_prog = train_layer_policy(
        label="FC3",
        model=model,
        policy=policy_fc3,
        physics=physics,
        optimizer=opt_fc3,
        train_loader=train_loader,
        num_batches=args.fc3_batches,
        initial_conductance=fc3_cond_noisy,
        active_layer=3,
        fc1_context=fc1_cond_noisy,
        fc2_context=fc2_cond_noisy,
        fc3_context=None,
        steps=args.fc3_steps,
        device=device,
    )

    eval_fc3 = evaluate_full(
        model=model,
        fc1_cond=fc1_cond_noisy,
        fc2_cond=fc2_cond_noisy,
        fc3_cond=fc3_cond_prog,
        loader=test_loader,
        max_batches=args.eval_batches,
        device=device,
    )
    print(f"after_FC3  loss={eval_fc3['loss']:.4f} acc={eval_fc3['accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Phase 2: Program FC2
    # ------------------------------------------------------------------
    fc2_shape = model.fc2_conductance_shape
    use_fc2_blocks = args.fc2_block_rows > 0 and args.fc2_block_cols > 0
    print(
        f"\nstage=program_FC2  steps={args.fc2_steps} batches={args.fc2_batches} "
        f"blocks={'yes' if use_fc2_blocks else 'full'}"
    )
    fc2_cond_current = fc2_cond_noisy.clone()

    if not use_fc2_blocks:
        policy_fc2 = make_policy(args.policy_type, args.policy_hidden_dim, args.policy_hidden_layers, device)
        opt_fc2 = torch.optim.Adam(policy_fc2.parameters(), lr=args.policy_lr)
        fc2_cond_current = train_layer_policy(
            label="FC2",
            model=model,
            policy=policy_fc2,
            physics=physics,
            optimizer=opt_fc2,
            train_loader=train_loader,
            num_batches=args.fc2_batches,
            initial_conductance=fc2_cond_current,
            active_layer=2,
            fc1_context=fc1_cond_noisy,
            fc2_context=None,
            fc3_context=fc3_cond_prog,
            steps=args.fc2_steps,
            device=device,
        )
    else:
        br, bc = args.fc2_block_rows, args.fc2_block_cols
        out_f, in_f = fc2_shape
        n_row_blocks = (out_f + br - 1) // br
        n_col_blocks = (in_f + bc - 1) // bc
        total_blocks = n_row_blocks * n_col_blocks
        block_idx = 0
        policy_fc2 = make_policy(args.policy_type, args.policy_hidden_dim, args.policy_hidden_layers, device)
        opt_fc2 = torch.optim.Adam(policy_fc2.parameters(), lr=args.policy_lr)
        for rs in range(0, out_f, br):
            for cs in range(0, in_f, bc):
                block_idx += 1
                mask = create_block_mask(fc2_shape, rs, cs, br, bc, device=device)
                fc2_cond_current = train_layer_policy(
                    label=f"FC2-block{block_idx}/{total_blocks}",
                    model=model,
                    policy=policy_fc2,
                    physics=physics,
                    optimizer=opt_fc2,
                    train_loader=train_loader,
                    num_batches=args.fc2_batches,
                    initial_conductance=fc2_cond_current,
                    active_layer=2,
                    fc1_context=fc1_cond_noisy,
                    fc2_context=None,
                    fc3_context=fc3_cond_prog,
                    steps=args.fc2_steps,
                    device=device,
                    programmable_mask=mask,
                    log_every=args.fc2_batches,
                )

    fc2_cond_prog = fc2_cond_current
    eval_fc2 = evaluate_full(
        model=model,
        fc1_cond=fc1_cond_noisy,
        fc2_cond=fc2_cond_prog,
        fc3_cond=fc3_cond_prog,
        loader=test_loader,
        max_batches=args.eval_batches,
        device=device,
    )
    print(f"after_FC2  loss={eval_fc2['loss']:.4f} acc={eval_fc2['accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Phase 3: Program FC1 (divide-and-conquer)
    # ------------------------------------------------------------------
    fc1_shape = model.fc1_conductance_shape
    br1, bc1 = args.fc1_block_rows, args.fc1_block_cols
    out_f1, in_f1 = fc1_shape
    n_row_blocks1 = (out_f1 + br1 - 1) // br1
    n_col_blocks1 = (in_f1 + bc1 - 1) // bc1
    total_blocks1 = n_row_blocks1 * n_col_blocks1
    print(
        f"\nstage=program_FC1  steps={args.fc1_steps} "
        f"batches_per_block={args.fc1_batches_per_block} "
        f"block_size={br1}x{bc1} total_blocks={total_blocks1}"
    )

    fc1_cond_current = fc1_cond_noisy.clone()
    policy_fc1 = make_policy(args.policy_type, args.policy_hidden_dim, args.policy_hidden_layers, device)
    opt_fc1 = torch.optim.Adam(policy_fc1.parameters(), lr=args.policy_lr)
    block_idx1 = 0
    for rs in range(0, out_f1, br1):
        for cs in range(0, in_f1, bc1):
            block_idx1 += 1
            mask = create_block_mask(fc1_shape, rs, cs, br1, bc1, device=device)
            selected = int(mask.sum().item())
            fc1_cond_current = train_layer_policy(
                label=f"FC1-block{block_idx1}/{total_blocks1}({selected}mem)",
                model=model,
                policy=policy_fc1,
                physics=physics,
                optimizer=opt_fc1,
                train_loader=train_loader,
                num_batches=args.fc1_batches_per_block,
                initial_conductance=fc1_cond_current,
                active_layer=1,
                fc1_context=None,
                fc2_context=fc2_cond_prog,
                fc3_context=fc3_cond_prog,
                steps=args.fc1_steps,
                device=device,
                programmable_mask=mask,
                log_every=args.fc1_batches_per_block,
            )

    fc1_cond_prog = fc1_cond_current
    eval_fc1 = evaluate_full(
        model=model,
        fc1_cond=fc1_cond_prog,
        fc2_cond=fc2_cond_prog,
        fc3_cond=fc3_cond_prog,
        loader=test_loader,
        max_batches=args.eval_batches,
        device=device,
    )
    print(f"after_FC1  loss={eval_fc1['loss']:.4f} acc={eval_fc1['accuracy']:.3f}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n=== Summary ===")
    print(f"trained    loss={trained_baseline['loss']:.4f}  acc={trained_baseline['accuracy']:.3f}")
    print(f"noisy      loss={noisy_baseline['loss']:.4f}  acc={noisy_baseline['accuracy']:.3f}")
    print(f"after_FC3  loss={eval_fc3['loss']:.4f}  acc={eval_fc3['accuracy']:.3f}  "
          f"delta={eval_fc3['accuracy'] - noisy_baseline['accuracy']:+.3f}")
    print(f"after_FC2  loss={eval_fc2['loss']:.4f}  acc={eval_fc2['accuracy']:.3f}  "
          f"delta={eval_fc2['accuracy'] - noisy_baseline['accuracy']:+.3f}")
    print(f"after_FC1  loss={eval_fc1['loss']:.4f}  acc={eval_fc1['accuracy']:.3f}  "
          f"delta={eval_fc1['accuracy'] - noisy_baseline['accuracy']:+.3f}")
    total_delta = eval_fc1['accuracy'] - noisy_baseline['accuracy']
    print(f"total_delta={total_delta:+.3f}")


if __name__ == "__main__":
    main()
