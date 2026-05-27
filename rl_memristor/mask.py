from __future__ import annotations

import torch
from torch import Tensor


def create_block_mask(
    shape: tuple[int, int],
    row_start: int,
    col_start: int,
    block_rows: int,
    block_cols: int,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    out_features, in_features = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    row_end = min(row_start + block_rows, out_features)
    col_end = min(col_start + block_cols, in_features)
    if row_end > row_start and col_end > col_start:
        mask[row_start:row_end, col_start:col_end] = True
    return mask


def create_fc1_programmable_mask(
    shape: tuple[int, int],
    *,
    mode: str = "full",
    row: int = 0,
    rows: int = 32,
    cols: int = 32,
    device: torch.device | str | None = None,
) -> Tensor:
    out_features, in_features = shape
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if mode == "full":
        mask[:, :] = True
    elif mode == "row":
        if not 0 <= row < out_features:
            raise ValueError(f"row must be in [0, {out_features}), got {row}.")
        mask[row, :] = True
    elif mode == "block":
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive for block mode.")
        mask[: min(rows, out_features), : min(cols, in_features)] = True
    else:
        raise ValueError(f"Unsupported programmable mask mode: {mode}.")
    return mask
