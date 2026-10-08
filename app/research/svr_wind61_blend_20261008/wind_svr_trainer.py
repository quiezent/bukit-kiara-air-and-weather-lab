"""One fixed CPU RBF SVR on the original basic31 plus frozen thirty issued-wind summaries."""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = ROOT / 'research/sequence_models_20261008/sequence_trainer.py'
_spec = importlib.util.spec_from_file_location('svr_original_channel_normalizer', ORIGINAL)
_original = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_original)

PARAMETERS = dict(kernel='rbf', C=10, epsilon=0.25, gamma='scale', tol=0.001,
                  shrinking=True, cache_size=512, max_iter=-1)


def specification():
    return {'version': 'one_fixed_wind61_rbf_svr_v1', 'parameters': PARAMETERS,
            'sklearnVersion': sklearn.__version__, 'device': 'cpu', 'threads': 1,
            'input': 'Original float32 context31 exactly plus frozen extra30; train-only61 means/std,61values then61flags, float32 transformed122' ,
            'target': 'Original float32 (actual-fresh)/20',
            'output': 'max(0,fresh+20*SVRhead)',
            'selection': 'One declared parameter set; no search or outcome-based adjustment'}


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ORIGINAL]}


class ContextSVR:
    def __init__(self):
        self.normalizer = None
        self.model = None
        self.training_metadata = {}

    def _arrays(self, bundle, indices, fit=False):
        values = np.asarray(bundle['context'][indices], dtype=np.float64)
        if fit:
            self.normalizer = _original._normalizer(values)
        mean, std = self.normalizer
        known = np.isfinite(values)
        scaled = np.where(known, (values-mean)/std, 0)
        joined = np.concatenate([scaled, (~known).astype(float)], axis=-1)
        if not np.isfinite(joined).all():
            raise ValueError('Nonfinite transformed context')
        return np.asarray(joined, dtype=np.float32)

    def fit(self, bundle, indices):
        started = time.monotonic()
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
            raise ValueError('Training indices must be nonempty and unique')
        features = self._arrays(bundle, indices, fit=True)
        actual = np.asarray(bundle['y'][indices], dtype=np.float64)
        fresh = np.asarray(bundle['fresh'][indices], dtype=np.float64)
        target = np.asarray((actual-fresh)/20, dtype=np.float32)
        if not np.isfinite(target).all():
            raise ValueError('Incomplete target/reference in training')
        self.model = SVR(**PARAMETERS)
        with threadpool_limits(limits=1):
            self.model.fit(features, target)
        self.training_metadata = {
            'trainingRows': len(indices), 'features': features.shape[1],
            'trainingSeconds': time.monotonic()-started, 'trainingDevice': 'cpu',
            'inferenceDevice': 'cpu', 'threads': 1, 'sklearnVersion': sklearn.__version__,
            'parameters': self.model.get_params(), 'fitStatus': int(self.model.fit_status_),
            'nIterations': int(self.model.n_iter_), 'nSupport': self.model.n_support_.tolist(),
            'supportVectors': len(self.model.support_), 'effectiveGamma': float(self.model._gamma),
            'trainingFeatureSha256': hashlib.sha256(features.tobytes()).hexdigest(),
            'trainingTargetSha256': hashlib.sha256(target.tobytes()).hexdigest(),
            'normalizerTrainingOnly': True}
        return self

    def predict(self, bundle, indices):
        indices = np.asarray(indices, dtype=np.int64)
        features = self._arrays(bundle, indices)
        with threadpool_limits(limits=1):
            head = self.model.predict(features)
        return np.maximum(0, np.asarray(bundle['fresh'][indices], dtype=np.float64)+20*head)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'normalizer': self.normalizer,
                     'specification': specification(), 'sourceHashes': source_hashes(),
                     'training_metadata': self.training_metadata}, path, compress=3)

    @classmethod
    def load(cls, path):
        saved = joblib.load(Path(path))
        if saved['sourceHashes'] != source_hashes():
            raise RuntimeError('Saved source differs')
        if saved['specification']['sklearnVersion'] != sklearn.__version__:
            raise RuntimeError('Saved sklearn version differs')
        obj = cls(); obj.model = saved['model']; obj.normalizer = saved['normalizer']
        obj.training_metadata = saved['training_metadata']
        return obj
