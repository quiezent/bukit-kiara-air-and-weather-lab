"""Offline fixed learned +90 concentration; inversion belongs to estimator."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

PARAMETERS = dict(loss="squared_error", learning_rate=.05, max_iter=100,
                  max_leaf_nodes=7, max_depth=3, min_samples_leaf=80,
                  l2_regularization=10., early_stopping=False,
                  random_state=20261009)


class ArrivalRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, reference_index=0):
        self.reference_index = reference_index

    def fit(self, matrix, physical_targets):
        x, y = np.asarray(matrix, float), np.asarray(physical_targets, float)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("invalid_training_shape")
        reference = x[:, self.reference_index]
        if not np.isfinite(reference).all() or not np.isfinite(y).all() or np.isinf(x).any():
            raise ValueError("invalid_training_reference_or_values")
        self.estimator_ = HistGradientBoostingRegressor(**PARAMETERS).fit(x, y-reference)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, matrix):
        x = np.asarray(matrix, float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or np.isinf(x).any():
            raise ValueError("invalid_prediction_shape_or_values")
        reference = x[:, self.reference_index]
        if not np.isfinite(reference).all():
            raise ValueError("missing_fresh_reference")
        return self.estimator_.predict(x) + reference
