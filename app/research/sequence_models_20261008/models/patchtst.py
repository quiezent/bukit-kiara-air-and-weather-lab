"""Fixed compact PatchTST-inspired forecaster with issued-weather conditioning.

Patching and the shared channel-independent encoder follow PatchTST's core
design (Nie et al., https://arxiv.org/abs/2211.14730). Pairing each sensor
channel with its availability mask and fusing weather/context after encoding
are explicit extensions, not a full reproduction of the paper's benchmark.
Normalization, causal input guards and training are owned by the caller.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


VERSION = "patchtst_weather_fusion_delta20_v1"
SPEC = {
    "version": VERSION,
    "source": "https://arxiv.org/abs/2211.14730",
    "interpretation": "Compact PatchTST-inspired channel-independent patch encoder; mask pairing and weather/context fusion are explicit extensions, not a canonical benchmark replication.",
    "input": {
        "past": "B x 96 x 6: three standardized sensor channels, then their three missing flags",
        "future": "B x 26 x 30: fifteen standardized originally issued weather/time/target channels, then their fifteen missing flags",
        "context": "B x 62: thirty-one standardized context features, then their thirty-one missing flags",
    },
    "patch": {"length": 12, "stride": 6, "endReplicationPadding": 6, "count": 16, "embedding": "Linear(24,32): value and missing flag paired per step"},
    "encoder": {"channelIndependent": True, "sharedWeights": True, "sensorChannels": 3, "learnedPosition": [16, 32], "layers": 2, "dModel": 32, "heads": 4, "feedForward": 64, "dropout": 0.1, "preLayerNorm": True, "activation": "ReLU"},
    "sensorReadout": "Shared Linear(16*32,32) per channel; concatenate three channel latents; Linear(96,32), ReLU",
    "futureBranch": "Flatten(26*future_dim), Linear(...,64), ReLU, Linear(64,32), ReLU",
    "contextBranch": "Linear(context_dim,32), ReLU",
    "fusionHead": "Concatenate three 32-dimensional branches; Linear(96,32), ReLU, Linear(32,1)",
    "output": "B normalized signed future delta; caller converts max(0, issue_reference + 20*output)",
    "training": "Caller-owned fixed policy: 30 epochs, seed 1749; no model-specific search or query-label tuning",
}


def specification():
    return deepcopy(SPEC)


class PatchTSTModel(nn.Module):
    """Patch and encode each sensor channel independently with shared weights."""

    def __init__(self, past_dim: int, future_dim: int, context_dim: int):
        super().__init__()
        if past_dim != 6:
            raise ValueError("PatchTST requires three sensor values followed by three missing flags (past_dim=6).")
        if future_dim <= 0 or context_dim <= 0:
            raise ValueError("future_dim and context_dim must be positive.")
        self.past_dim = past_dim
        self.future_dim = future_dim
        self.context_dim = context_dim
        self.patch_embedding = nn.Linear(24, 32)
        self.position = nn.Parameter(torch.empty(1, 16, 32))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.channel_readout = nn.Linear(16 * 32, 32)
        self.sensor_fusion = nn.Sequential(nn.Linear(3 * 32, 32), nn.ReLU())
        self.future_branch = nn.Sequential(
            nn.Linear(26 * future_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.context_branch = nn.Sequential(nn.Linear(context_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(3 * 32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, past: torch.Tensor, future: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if past.ndim != 3 or past.shape[1:] != (96, self.past_dim):
            raise ValueError("past must have shape B x 96 x 6.")
        if future.ndim != 3 or future.shape[1:] != (26, self.future_dim):
            raise ValueError("future must have shape B x 26 x future_dim.")
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError("context must have shape B x context_dim.")
        if past.shape[0] != future.shape[0] or past.shape[0] != context.shape[0]:
            raise ValueError("All inputs must share the batch dimension.")
        batch = past.shape[0]
        # B,T,C,2 -> B,C,T,2; each channel receives only its own value/mask.
        paired = torch.stack((past[..., :3], past[..., 3:]), dim=-1).permute(0, 2, 1, 3)
        paired = torch.cat((paired, paired[:, :, -1:, :].expand(-1, -1, 6, -1)), dim=2)
        # unfold appends the patch-length axis: B,C,N,2,P -> B,C,N,P,2.
        patches = paired.unfold(2, size=12, step=6).permute(0, 1, 2, 4, 3)
        patches = patches.reshape(batch * 3, 16, 24)
        encoded = self.encoder(self.patch_embedding(patches) + self.position)
        channels = self.channel_readout(encoded.reshape(batch * 3, 16 * 32))
        sensor = self.sensor_fusion(channels.reshape(batch, 3 * 32))
        weather = self.future_branch(future.reshape(batch, 26 * self.future_dim))
        static = self.context_branch(context)
        return self.head(torch.cat((sensor, weather, static), dim=-1)).squeeze(-1)
