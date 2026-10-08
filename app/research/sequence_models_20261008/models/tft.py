"""Compact TFT point-forecast adaptation for the frozen TTDI sequence trial.

The caller owns train-only normalization and chronological information cutoffs.
This module neither fits a model nor reads real data. It preserves the TFT
building blocks described by Lim et al., while replacing per-horizon quantiles
with one context-conditioned pooled scalar: normalized signed PM2.5 change.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VERSION = "tft_point_delta20_sequence_v1"
SPEC = {
    "version": VERSION,
    "reference": "https://arxiv.org/abs/1912.09363",
    "referenceImplementation": "https://github.com/google-research/google-research/tree/master/tft",
    "past": "B x 96 x 6: three standardized numeric variables followed by their three missing flags",
    "future": "B x 26 x 30: fifteen standardized originally issued weather/time/target variables followed by their fifteen missing flags",
    "context": "B x 62: basic31 standardized issue-time features followed by their missing flags",
    "variableEmbedding": "Separate learned Linear(2,32) for each numeric-plus-missing pair; separate past and future variable GRNs",
    "dModel": 32,
    "hiddenSize": 32,
    "dropout": 0.1,
    "attentionHeads": 4,
    "attentionValue": "One shared 32-to-8 value projection, four distinct query/key projections, average heads, 8-to-32 output projection",
    "attentionMask": "Causal upper-triangle mask over concatenated past plus future positions",
    "contextEncoding": "Issue-time context GRN plus distinct GRNs for variable selection, static enrichment, encoder initial hidden and cell states, and pooling",
    "variableSelection": "Context-conditioned GRN logits followed by softmax, per-variable GRNs and weighted sum",
    "localProcessing": "One-layer encoder LSTM and one-layer decoder LSTM initialized from encoder final states",
    "fusion": "Gated local residual; context enrichment GRN; interpretable temporal attention; gated attention residual; positionwise GRN; gated residual to local features",
    "pointHead": "Context-conditioned learned attention pool of 26 decoder representations; concatenated context; Linear64-to-32, ELU, dropout, Linear32-to-1",
    "adaptations": [
        "Context is issue-time covariates rather than entity-static metadata; the model never treats it as future observations.",
        "A GRN encodes the context vector as a whole rather than performing a second static variable-selection network.",
        "Scalar exact-target pooling and a point loss replace the canonical TFT multi-horizon quantile head.",
        "The shared trainer uses SmoothL1 on normalized signed change; this is not a full TFT benchmark replication.",
    ],
    "trainingByCaller": {"optimizer": "AdamW", "lr": 0.001, "batchSize": 128, "epochs": 30, "seed": 1749},
}


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.value = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.value(x) * torch.sigmoid(self.gate(x))


class GateAddNorm(nn.Module):
    def __init__(self, size: int, dropout: float = 0.1):
        super().__init__()
        self.glu = GatedLinearUnit(size, size, dropout)
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + self.glu(x))


class GatedResidualNetwork(nn.Module):
    """TFT GRN: ELU hidden transform, optional context, GLU, skip and norm."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 context_size: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.context_layer = (nn.Linear(context_size, hidden_size, bias=False)
                              if context_size is not None else None)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        self.skip = (nn.Identity() if input_size == output_size
                     else nn.Linear(input_size, output_size))
        self.gate_norm = GateAddNorm(output_size, dropout)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.input_layer(x)
        if self.context_layer is not None:
            if context is None:
                raise ValueError("This GRN requires context")
            while context.ndim < hidden.ndim:
                context = context.unsqueeze(1)
            hidden = hidden + self.context_layer(context)
        hidden = self.hidden_layer(F.elu(hidden))
        return self.gate_norm(hidden, self.skip(x))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, variables: int, size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.variables = variables
        self.embeddings = nn.ModuleList(nn.Linear(2, size) for _ in range(variables))
        self.variable_grns = nn.ModuleList(
            GatedResidualNetwork(size, size, size, dropout=dropout)
            for _ in range(variables)
        )
        self.weight_grn = GatedResidualNetwork(
            variables * size, size, variables, context_size=size, dropout=dropout)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, flags = x.split(self.variables, dim=-1)
        pairs = torch.stack((values, flags), dim=-1)
        embeddings = [layer(pairs[..., index, :])
                      for index, layer in enumerate(self.embeddings)]
        logits = self.weight_grn(torch.cat(embeddings, dim=-1), context)
        weights = torch.softmax(logits, dim=-1)
        transformed = torch.stack([grn(value) for grn, value in
                                   zip(self.variable_grns, embeddings)], dim=-2)
        selected = (transformed * weights.unsqueeze(-1)).sum(dim=-2)
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """TFT attention uses shared values and head averaging, not concatenation."""
    def __init__(self, size: int = 32, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if size % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_size = size // heads
        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.shared_value = nn.Linear(size, self.head_size)
        self.output = nn.Linear(self.head_size, size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = sequence.shape
        query = self.query(sequence).view(batch, length, self.heads,
                                         self.head_size).transpose(1, 2)
        key = self.key(sequence).view(batch, length, self.heads,
                                     self.head_size).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(mask.view(1, 1, length, length), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        shared_value = self.shared_value(sequence).unsqueeze(1)
        attended = torch.matmul(self.dropout(weights), shared_value).mean(dim=1)
        return self.output(attended), weights.mean(dim=1)


class TFTModel(nn.Module):
    def __init__(self, past_dim: int, future_dim: int, context_dim: int):
        super().__init__()
        if (past_dim, future_dim, context_dim) != (6, 30, 62):
            raise ValueError("The frozen TFT recipe expects input dimensions 6, 30, 62")
        size, dropout = 32, 0.1
        self.context_grn = GatedResidualNetwork(context_dim, size, size, dropout=dropout)
        self.selection_context = GatedResidualNetwork(size, size, size, dropout=dropout)
        self.enrichment_context = GatedResidualNetwork(size, size, size, dropout=dropout)
        self.initial_hidden = GatedResidualNetwork(size, size, size, dropout=dropout)
        self.initial_cell = GatedResidualNetwork(size, size, size, dropout=dropout)
        self.pool_context = GatedResidualNetwork(size, size, size, dropout=dropout)
        self.past_selection = VariableSelectionNetwork(3, size, dropout)
        self.future_selection = VariableSelectionNetwork(15, size, dropout)
        self.encoder = nn.LSTM(size, size, batch_first=True)
        self.decoder = nn.LSTM(size, size, batch_first=True)
        self.local_gate = GateAddNorm(size, dropout)
        self.enrichment = GatedResidualNetwork(size, size, size,
                                              context_size=size, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(size, 4, dropout)
        self.attention_gate = GateAddNorm(size, dropout)
        self.feedforward = GatedResidualNetwork(size, size, size, dropout=dropout)
        self.output_gate = GateAddNorm(size, dropout)
        self.pool_score = nn.Linear(size, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(2 * size, size), nn.ELU(),
                                  nn.Dropout(dropout), nn.Linear(size, 1))
        self.register_buffer("causal_mask", torch.triu(
            torch.ones(122, 122, dtype=torch.bool), diagonal=1), persistent=False)

    def _forward(self, past: torch.Tensor, future: torch.Tensor,
                 context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if past.ndim != 3 or tuple(past.shape[1:]) != (96, 6):
            raise ValueError("past must have shape B x 96 x 6")
        if future.ndim != 3 or tuple(future.shape[1:]) != (26, 30):
            raise ValueError("future must have shape B x 26 x 30")
        if context.ndim != 2 or context.shape[1] != 62:
            raise ValueError("context must have shape B x 62")
        if past.shape[0] != future.shape[0] or past.shape[0] != context.shape[0]:
            raise ValueError("All inputs must share the batch dimension")
        static = self.context_grn(context)
        selection_context = self.selection_context(static)
        past_selected, past_weights = self.past_selection(past, selection_context)
        future_selected, future_weights = self.future_selection(future, selection_context)
        initial_state = (self.initial_hidden(static).unsqueeze(0),
                         self.initial_cell(static).unsqueeze(0))
        encoded, encoder_state = self.encoder(past_selected, initial_state)
        decoded, _ = self.decoder(future_selected, encoder_state)
        selected = torch.cat((past_selected, future_selected), dim=1)
        local = self.local_gate(torch.cat((encoded, decoded), dim=1), selected)
        enriched = self.enrichment(local, self.enrichment_context(static))
        attended, attention_weights = self.attention(enriched, self.causal_mask)
        fused = self.attention_gate(attended, enriched)
        final = self.output_gate(self.feedforward(fused), local)
        future_final = final[:, 96:, :]
        pool_query = self.pool_context(static).unsqueeze(1)
        pool_weights = torch.softmax(self.pool_score(torch.tanh(future_final + pool_query)), dim=1)
        pooled = (future_final * pool_weights).sum(dim=1)
        prediction = self.head(torch.cat((pooled, static), dim=-1)).squeeze(-1)
        return prediction, {
            "pastVariableWeights": past_weights,
            "futureVariableWeights": future_weights,
            "temporalAttentionWeights": attention_weights,
            "decoderPoolWeights": pool_weights.squeeze(-1),
        }

    def forward(self, past: torch.Tensor, future: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        return self._forward(past, future, context)[0]

    def forward_with_weights(self, past: torch.Tensor, future: torch.Tensor,
                             context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Explicit diagnostic call; ordinary forward does not retain batch tensors."""
        return self._forward(past, future, context)


def specification() -> dict:
    return {**SPEC, "codeSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def synthetic_check() -> dict:
    """CPU-only plumbing check, separate from any claim of predictive skill."""
    torch.manual_seed(1749)
    torch.set_num_threads(1)
    model = TFTModel(6, 30, 62)
    past = torch.randn(2, 96, 6)
    future = torch.randn(2, 26, 30)
    context = torch.randn(2, 62)
    past[..., 3:] = torch.randint(0, 2, past[..., 3:].shape).float()
    future[..., 15:] = torch.randint(0, 2, future[..., 15:].shape).float()
    context[..., 31:] = torch.randint(0, 2, context[..., 31:].shape).float()
    model.train()
    output = model(past, future, context)
    loss = F.smooth_l1_loss(output, torch.tensor([0.3, -0.7]))
    loss.backward()
    if output.shape != (2,) or not torch.isfinite(output).all():
        raise AssertionError("Invalid output shape or nonfinite output")
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
        raise AssertionError("Missing or nonfinite gradients")
    model.eval()
    with torch.no_grad():
        baseline, weights = model.forward_with_weights(past, future, context)
        changed_past = past.clone(); changed_past[:, -8:, 0] += 2.0
        changed_future = future.clone(); changed_future[..., 0] += 2.0
        changed_context = context.clone(); changed_context[:, 0] += 2.0
        changes = {
            "past": float((model(changed_past, future, context) - baseline).abs().max()),
            "future": float((model(past, changed_future, context) - baseline).abs().max()),
            "context": float((model(past, future, changed_context) - baseline).abs().max()),
        }
        if any(value <= 1e-8 for value in changes.values()):
            raise AssertionError("One input branch has no effect on output")
        errors = {name: float((weight.sum(dim=-1) - 1).abs().max())
                  for name, weight in weights.items()}
        if any(value > 1e-5 for value in errors.values()):
            raise AssertionError("Selection/attention/pooling weights do not sum to one")
        causal_leak = float(weights["temporalAttentionWeights"][:, model.causal_mask].abs().max())
        if causal_leak != 0:
            raise AssertionError("Attention is not causal")
    return {"device": "cpu", "outputShape": list(output.shape), "loss": float(loss.detach()),
            "parameters": sum(param.numel() for param in model.parameters()),
            "finiteGradientTensors": len(gradients), "inputVariationMaxAbs": changes,
            "weightSumMaxAbsError": errors, "futureAttentionLeakMaxAbs": causal_leak,
            "predictiveSkillTest": False}


if __name__ == "__main__":
    print(json.dumps({"specification": specification(), "syntheticCheck": synthetic_check()}, indent=2))
