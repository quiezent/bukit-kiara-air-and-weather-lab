"""Frozen small TCN and LSTM forecasting candidates, with shared input branches.

These modules contain no fitting or preprocessing. The caller supplies causal,
training-standardized values and missing flags. Output is a signed PM2.5 target
change divided by 20, not an absolute concentration or a reconstruction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PAST_STEPS = 96
FUTURE_STEPS = 26
HIDDEN = 32
DILATIONS = (1, 2, 4, 8, 16)


class _CausalConv1d(nn.Module):
    """Left padding only; output at t never uses input after t."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, dilation=dilation,
            padding=0,
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class _CausalResidualBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, HIDDEN, dilation)
        self.conv2 = _CausalConv1d(HIDDEN, HIDDEN, dilation)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.residual = (
            nn.Conv1d(in_channels, HIDDEN, kernel_size=1)
            if in_channels != HIDDEN else nn.Identity()
        )

    def forward(self, values: Tensor) -> Tensor:
        hidden = self.dropout1(F.relu(self.conv1(values)))
        hidden = self.dropout2(F.relu(self.conv2(hidden)))
        return F.relu(hidden + self.residual(values))


class _ForecastModel(nn.Module):
    def __init__(self, past_dim: int, future_dim: int, context_dim: int):
        super().__init__()
        if min(past_dim, future_dim, context_dim) <= 0:
            raise ValueError("Input dimensions must be positive.")
        self.past_dim = int(past_dim)
        self.future_dim = int(future_dim)
        self.context_dim = int(context_dim)
        self.future_branch = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(FUTURE_STEPS * self.future_dim, 64), nn.ReLU(),
            nn.Linear(64, HIDDEN), nn.ReLU(),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(self.context_dim, HIDDEN), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(3 * HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def _check_inputs(self, past: Tensor, future: Tensor, context: Tensor) -> None:
        if past.ndim != 3 or tuple(past.shape[1:]) != (PAST_STEPS, self.past_dim):
            raise ValueError(f"past must have shape (B, {PAST_STEPS}, {self.past_dim}).")
        if future.ndim != 3 or tuple(future.shape[1:]) != (FUTURE_STEPS, self.future_dim):
            raise ValueError(f"future must have shape (B, {FUTURE_STEPS}, {self.future_dim}).")
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(f"context must have shape (B, {self.context_dim}).")
        if not (past.shape[0] == future.shape[0] == context.shape[0]):
            raise ValueError("All input branches must have the same batch size.")

    def encode_past(self, past: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, past: Tensor, future: Tensor, context: Tensor) -> Tensor:
        self._check_inputs(past, future, context)
        joined = torch.cat(
            (self.encode_past(past), self.future_branch(future),
             self.context_branch(context)), dim=1,
        )
        return self.head(joined).squeeze(-1)


class TCNModel(_ForecastModel):
    """Five causal residual blocks; final observed step is the past latent."""

    def __init__(self, past_dim: int, future_dim: int, context_dim: int):
        super().__init__(past_dim, future_dim, context_dim)
        self.past_encoder = nn.Sequential(*[
            _CausalResidualBlock(self.past_dim if index == 0 else HIDDEN, dilation)
            for index, dilation in enumerate(DILATIONS)
        ])

    def encode_past(self, past: Tensor) -> Tensor:
        return self.past_encoder(past.transpose(1, 2))[:, :, -1]


class LSTMModel(_ForecastModel):
    """One unidirectional LSTM layer; its last hidden state is the past latent."""

    def __init__(self, past_dim: int, future_dim: int, context_dim: int):
        super().__init__(past_dim, future_dim, context_dim)
        self.past_encoder = nn.LSTM(
            input_size=self.past_dim, hidden_size=HIDDEN, num_layers=1,
            batch_first=True, dropout=0.0, bidirectional=False,
        )

    def encode_past(self, past: Tensor) -> Tensor:
        _, (last_hidden, _) = self.past_encoder(past)
        return last_hidden[-1]


def specification() -> dict[str, Any]:
    """Return the declared recipe; training belongs to the shared runner."""
    return {
        "version": "tcn_lstm_sequence_delta20_v1",
        "input_contract": {
            "past": ["B", PAST_STEPS, 6],
            "past_channels": ["pm25_z", "temperature_z", "humidity_z",
                              "pm25_missing", "temperature_missing", "humidity_missing"],
            "future": ["B", FUTURE_STEPS, 30],
            "future_channels": "15 training-standardized channels followed by 15 missing flags",
            "context": ["B", 62],
            "context_channels": "basic31 training-standardized channels followed by 31 missing flags",
            "output": ["B"],
            "output_semantics": "signed exact-target PM2.5 change divided by 20",
            "scaling": "caller fits scaling on completed training inputs only",
        },
        "tcn": {
            "channels": HIDDEN, "kernel_size": 3, "dilations": list(DILATIONS),
            "blocks": 5, "convolutions_per_block": 2, "dropout": 0.1,
            "padding": "left only, 2*dilation for each convolution",
            "residual": "1x1 projection in first block, identity thereafter",
            "activation": "ReLU after each convolution and after residual addition",
            "receptive_field_steps": 125,
            "past_latent": "last observed step",
        },
        "lstm": {
            "hidden_size": HIDDEN, "layers": 1, "bidirectional": False,
            "dropout": 0.0, "batch_first": True,
            "past_latent": "last hidden state",
        },
        "shared_future_branch": ["flatten 26*future_dim", 64, "ReLU", HIDDEN, "ReLU"],
        "shared_context_branch": ["context_dim", HIDDEN, "ReLU"],
        "shared_head": [3 * HIDDEN, HIDDEN, "ReLU", 1],
        "parameter_initialization": "PyTorch layer defaults; caller sets seed before construction",
        "training": {
            "owner": "shared runner", "optimizer": "AdamW", "learning_rate": 0.001,
            "weight_decay": 0.001, "batch_size": 128, "epochs": 30,
            "seed": 1749, "loss": "SmoothL1 on delta/20",
        },
    }
