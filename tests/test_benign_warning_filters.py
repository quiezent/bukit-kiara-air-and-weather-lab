"""The service silences two known-benign library warnings and nothing else.

Both warnings come from the real library code paths. No service, database,
network or live model is touched.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import types
import unittest
import warnings

import numpy as np
import threadpoolctl
from sklearn.linear_model import LogisticRegression

import BukitKiara_Dashboard as dashboard


def emit_openmp_warning():
    """threadpoolctl's own check, reporting both OpenMP runtimes loaded."""
    controller = types.SimpleNamespace(lib_controllers=[
        types.SimpleNamespace(prefix="libomp"), types.SimpleNamespace(prefix="libiomp")])
    threadpoolctl.ThreadpoolController._warn_if_incompatible_openmp(controller)


def fit_multinomial():
    x = np.array([[0.0], [1.0], [2.0], [0.2], [1.2], [2.2]])
    LogisticRegression(multi_class="multinomial", max_iter=200).fit(x, [0, 1, 2, 0, 1, 2])


def recorded(emit, silenced):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if silenced:
            dashboard.silence_known_benign_warnings()
        emit()
    return caught


class KnownBenignWarningFilterTests(unittest.TestCase):
    def test_threadpoolctl_openmp_warning_is_filtered(self):
        loud = recorded(emit_openmp_warning, silenced=False)
        self.assertEqual([w.category for w in loud], [RuntimeWarning])
        self.assertEqual(recorded(emit_openmp_warning, silenced=True), [])

    def test_sklearn_multi_class_deprecation_is_filtered(self):
        def multi_class(caught):
            return [w for w in caught if w.category is FutureWarning
                    and "'multi_class' was deprecated" in str(w.message)]
        self.assertEqual(len(multi_class(recorded(fit_multinomial, silenced=False))), 1)
        self.assertEqual(multi_class(recorded(fit_multinomial, silenced=True)), [])

    def test_fit_warning_metadata_still_records_multi_class(self):
        # arrival_logistic_model records fit warnings under simplefilter("always").
        with warnings.catch_warnings():
            dashboard.silence_known_benign_warnings()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_multinomial()
        self.assertTrue(any("'multi_class' was deprecated" in str(w.message) for w in caught))

    def test_other_runtime_warnings_still_reach_the_log(self):
        openmp_text = str(recorded(emit_openmp_warning, silenced=False)[0].message)

        def emit():
            warnings.warn("unrelated runtime problem", RuntimeWarning)
            warnings.warn_explicit("another threadpoolctl problem", RuntimeWarning,
                                   "threadpoolctl.py", 1, module="threadpoolctl")
            warnings.warn_explicit(openmp_text, RuntimeWarning, "other.py", 1, module="other_module")

        caught = recorded(emit, silenced=True)
        self.assertEqual([str(w.message) for w in caught],
                         ["unrelated runtime problem", "another threadpoolctl problem", openmp_text])
        self.assertTrue(all(w.category is RuntimeWarning for w in caught))


if __name__ == "__main__":
    unittest.main()
