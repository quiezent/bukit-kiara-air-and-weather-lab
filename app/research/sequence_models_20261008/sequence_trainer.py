"""Fixed offline training for four weather-conditioned sequence architectures."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SEED = 1749
EPOCHS = 30
BATCH = 128
VARIANTS = ("tcn", "lstm", "patchtst", "tft")


def _module(name):
    path = HERE / "models" / (name + ".py")
    spec = importlib.util.spec_from_file_location("ttdi_sequence_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(variant, past_dim=6, future_dim=30, context_dim=62):
    if variant in ("tcn", "lstm"):
        module = _module("tcn_lstm")
        cls = module.TCNModel if variant == "tcn" else module.LSTMModel
    elif variant == "patchtst":
        cls = _module("patchtst").PatchTSTModel
    elif variant == "tft":
        cls = _module("tft").TFTModel
    else:
        raise ValueError("Unknown frozen architecture: " + variant)
    return cls(past_dim, future_dim, context_dim)


def specification():
    return {
        "version": "ttdi_four_small_weather_sequence_models_v1",
        "variants": list(VARIANTS),
        "seed": SEED, "epochs": EPOCHS, "batchSize": BATCH,
        "optimizer": "AdamW", "learningRate": 0.001, "weightDecay": 0.001,
        "loss": "SmoothL1 beta1 on (exact absolute target - issue fresh)/20",
        "output": "max(0, original candidate fresh +20*neural delta head)",
        "past": "96 covered/masked15minute steps, PM2.5/temp/RH, original published-watermark bounded",
        "future": "26 original-issued weather steps at issue+j hours;12 met channels plus3 deterministic clock/target markers",
        "context": "Existing guarded basic31; identical raw inputs and retained cases for all4 models",
        "preprocessing": "Train-only observed channel means/std; allmissing mean0/std1; tiny std<1e-6=>1; unknown standardized0 plus missing flags",
        "shuffle": "Local numpy seed1749, one fixed permutation per epoch; no data augmentation",
        "devices": "CUDA training, CPU inference threads1; no device or epoch selection using scores",
        "determinism": "Torch manual/cuda seed1749; deterministic algorithms; cuDNN deterministic, benchmarkFalse; CUBLAS workspace4096:8",
        "selection": "No parameter sweep, outcome early stopping or test-fitted normalization; all4 reported",
        "architectureAdaptations": "PatchTST has explicit shared-data weather/context fusion; TFT uses direct scalar mean head/context-conditioned decoder pooling instead of quantile outputs",
    }


def source_hashes():
    files = [Path(__file__)] + [HERE / "models" / (n + ".py")
                               for n in ("tcn_lstm", "patchtst", "tft")]
    return {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def _normalizer(values):
    value = np.asarray(values, dtype=np.float64)
    axes = tuple(range(value.ndim - 1))
    known = np.isfinite(value)
    count = known.sum(axis=axes)
    mean = np.divide(np.where(known, value, 0).sum(axis=axes), count,
                     out=np.zeros(value.shape[-1]), where=count > 0)
    diff = np.where(known, value - mean, 0)
    variance = np.divide((diff * diff).sum(axis=axes), count,
                         out=np.zeros(value.shape[-1]), where=count > 0)
    std = np.sqrt(variance)
    std[(count == 0) | (std < 1e-6)] = 1
    return mean, std


class SequenceRegressor:
    def __init__(self, variant, device="cuda"):
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.device = variant, str(device)
        self.normalizers = {}
        self.model = None
        self.training_metadata = {}

    def _arrays(self, bundle, indices, fit=False):
        arrays = []
        for key in ("past", "future", "context"):
            values = np.asarray(bundle[key][indices], dtype=np.float64)
            if fit:
                self.normalizers[key] = _normalizer(values)
            mean, std = self.normalizers[key]
            known = np.isfinite(values)
            scaled = np.where(known, (values - mean) / std, 0)
            joined = np.concatenate([scaled, (~known).astype(float)], axis=-1)
            if not np.isfinite(joined).all():
                raise ValueError("Nonfinite transformed " + key)
            arrays.append(np.asarray(joined, dtype=np.float32))
        return arrays

    def fit(self, bundle, indices):
        started = time.monotonic()
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Declared CUDA training unavailable")
            torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
            raise ValueError("Training indices must be nonempty and unique")
        past, future, context = self._arrays(bundle, indices, fit=True)
        absolute_y = np.asarray(bundle["y"][indices], dtype=np.float64)
        fresh = np.asarray(bundle["fresh"][indices], dtype=np.float64)
        target = np.asarray((absolute_y - fresh) / 20, dtype=np.float32)
        if not np.isfinite(target).all():
            raise ValueError("Incomplete target/reference in training")
        self.model = make_model(self.variant, past.shape[-1], future.shape[-1],
                                context.shape[-1]).to(self.device)
        initial_digest = hashlib.sha256(b"".join(
            p.detach().cpu().numpy().tobytes() for p in self.model.parameters())).hexdigest()
        tensors = [torch.as_tensor(a, device=self.device)
                   for a in (past, future, context, target)]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001,
                                     weight_decay=0.001)
        criterion = nn.SmoothL1Loss(beta=1)
        rng = np.random.default_rng(SEED)
        losses = []
        self.model.train()
        for epoch in range(EPOCHS):
            permutation = rng.permutation(len(indices))
            total = 0.0
            for first in range(0, len(indices), BATCH):
                take = torch.as_tensor(permutation[first:first+BATCH], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(*(t[take] for t in tensors[:3])).reshape(-1)
                loss = criterion(prediction, tensors[3][take])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; no outcome-based fallback")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(take)
            losses.append(total / len(indices))
        self.model.eval()
        with torch.no_grad():
            selected = [t[:min(8, len(indices))] for t in tensors[:3]]
            fitted_output = self.model(*selected).detach().cpu().numpy()
        self.model.to("cpu")
        with torch.no_grad():
            cpu_output = self.model(*(t.detach().cpu() for t in selected)).numpy()
        if not np.isfinite(cpu_output).all():
            raise FloatingPointError("Nonfinite CPU inference")
        max_difference = float(np.max(np.abs(cpu_output - fitted_output)))
        if not np.allclose(cpu_output, fitted_output, atol=1e-4, rtol=1e-4):
            raise AssertionError("CPU/CUDA normalized-head disagreement " + str(max_difference))
        final_digest = hashlib.sha256(b"".join(
            p.detach().numpy().tobytes() for p in self.model.parameters())).hexdigest()
        self.training_metadata = {
            "variant": self.variant, "trainingRows": len(indices),
            "parameterCount": sum(p.numel() for p in self.model.parameters()),
            "trainingSeconds": time.monotonic() - started,
            "trainingDevice": self.device, "inferenceDevice": "cpu",
            "epochs": EPOCHS, "epochLosses": losses,
            "initialStateSha256": initial_digest, "fittedStateSha256": final_digest,
            "parametersChanged": initial_digest != final_digest,
            "cpuCudaNormalizedHeadMaximumDifference": max_difference,
            "torchVersion": torch.__version__,
            "deviceName": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
            "normalizerTrainingOnly": True,
        }
        del tensors, optimizer, selected
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return self

    def predict(self, bundle, indices):
        if self.model is None:
            raise RuntimeError("Fit or load saved artifact first")
        torch.set_num_threads(1)
        self.model.eval()
        indices = np.asarray(indices, dtype=np.int64)
        arrays = self._arrays(bundle, indices)
        predictions = []
        with torch.no_grad():
            for first in range(0, len(indices), BATCH):
                inputs = [torch.from_numpy(a[first:first+BATCH]) for a in arrays]
                predictions.append(self.model(*inputs).reshape(-1).numpy())
        head = np.concatenate(predictions) if predictions else np.empty(0)
        fresh = np.asarray(bundle["fresh"][indices], dtype=np.float64)
        return np.maximum(0, fresh + 20 * head.astype(np.float64))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"variant": self.variant, "specification": specification(),
                    "sourceHashes": source_hashes(), "normalizers": self.normalizers,
                    "state_dict": self.model.state_dict(),
                    "inputDimensions": [6, 30, 62],
                    "training_metadata": self.training_metadata}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(Path(path), map_location="cpu", weights_only=False)
        if saved["sourceHashes"] != source_hashes():
            raise RuntimeError("Saved architecture/trainer source differs")
        obj = cls(saved["variant"], device="cpu")
        obj.normalizers = saved["normalizers"]
        obj.model = make_model(saved["variant"], *saved["inputDimensions"])
        obj.model.load_state_dict(saved["state_dict"])
        obj.model.eval()
        obj.training_metadata = saved["training_metadata"]
        return obj
