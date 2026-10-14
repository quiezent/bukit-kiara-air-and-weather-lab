"""Cached-only publication must not reconstruct, fit, or block on a fit."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from datetime import datetime,timedelta,timezone
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import weather_session_forecast as forecast
import weather_session_model as learner

ISSUE=int(datetime(2026,10,8,12,tzinfo=timezone(timedelta(hours=8))).timestamp())
START=ISSUE+21*3600
WINDOWS={"morning":{"startEpoch":START,"endEpoch":START+7200}}
ROWS=[{"epoch":ISSUE-30,"pm02":100.}]


class CachedOnlyTests(unittest.TestCase):
    def setUp(self):
        with forecast._LOCK:
            forecast._DAILY_CACHE.clear()
            forecast._REFRESH_STATUS.clear()
            forecast._FIT_CACHE.clear()

    def test_cache_miss_has_no_reconstruction_or_legacy_fit(self):
        with patch.object(forecast,"_trained_model",side_effect=AssertionError("no inline fit")), \
             patch.object(forecast.legacy,"predict_windows",side_effect=AssertionError("no fallback fit")), \
             patch.object(forecast.rain,"load_runs",side_effect=AssertionError("no archive load on cold miss")), \
             patch.object(forecast.causal,"strict_frame",side_effect=AssertionError("no reconstruction on cold miss")):
            result=forecast.predict_windows("unused",ROWS,WINDOWS,ISSUE,allow_fit=False)["morning"]
        self.assertFalse(result["available"])
        self.assertIsNone(result["mean"])
        self.assertEqual(result["startEpoch"],START)
        self.assertFalse(result["candidateForecast"]["available"])
        self.assertEqual(result["weatherSessionModel"]["fitStatus"]["state"],"background_cache_pending")

    def test_exact_daily_identity_and_fit_receipt_are_required(self):
        cutoff=forecast.training_cutoff(ISSUE)
        key=forecast._daily_key("unused",cutoff)
        fitted={"trainingCutoffEpoch":cutoff,"trainingLatestTargetEndEpoch":cutoff-900,
                "fittedAtEpoch":ISSUE-30,"estimator":object()}
        with forecast._LOCK:
            forecast._DAILY_CACHE[key]=fitted
        self.assertIs(forecast._cached_model("unused",ISSUE),fitted)
        self.assertIsNone(forecast._cached_model("different-db",ISSUE))
        self.assertIsNone(forecast._cached_model("unused",ISSUE+86400))
        fitted["fittedAtEpoch"]=ISSUE+1
        self.assertIsNone(forecast._cached_model("unused",ISSUE))

    def test_cached_query_failure_does_not_route_to_legacy_fit(self):
        cutoff=forecast.training_cutoff(ISSUE)
        with forecast._LOCK:
            forecast._DAILY_CACHE[forecast._daily_key("unused",cutoff)]={
                "trainingCutoffEpoch":cutoff,"trainingLatestTargetEndEpoch":cutoff-900,
                "fittedAtEpoch":ISSUE-30,"estimator":object()}
        with patch.object(forecast.causal,"strict_frame",return_value=pd.DataFrame()), \
             patch.object(forecast.legacy,"predict_windows",side_effect=AssertionError("no fallback fit")), \
             patch.object(forecast,"_trained_model",side_effect=AssertionError("no inline fit")):
            result=forecast.predict_windows("unused",ROWS,WINDOWS,ISSUE,allow_fit=False)["morning"]
        self.assertFalse(result["available"])
        self.assertIsNone(result["mean"])
        self.assertFalse(result["weatherSessionModel"]["applied"])

    def test_background_fit_releases_cache_lock_and_does_not_block_publication(self):
        cutoff=forecast.training_cutoff(ISSUE)
        complete=np.repeat(cutoff-np.arange(1,13)*86400,10)
        snapshot={"valid":np.ones(120,dtype=bool),"labels":np.ones(120),
                  "complete":complete,"issues":complete-7200,"x":pd.DataFrame({"x":np.ones(120)})}
        entered,release=threading.Event(),threading.Event()
        def fake_fit(*args):
            entered.set()
            if not release.wait(3):raise AssertionError("test release timed out")
            return object()
        result=[]
        with patch.object(learner,"build_dataset",return_value=snapshot), \
             patch.object(learner,"fit_model",side_effect=fake_fit), \
             patch.object(forecast.legacy,"_raw_runs",return_value=([],"revision")):
            worker=threading.Thread(target=lambda:result.append(forecast.refresh_model("unused",ISSUE,rows=[])))
            worker.start()
            self.assertTrue(entered.wait(1))
            try:
                before=time.perf_counter()
                point=forecast.predict_windows("unused",ROWS,WINDOWS,ISSUE,allow_fit=False)["morning"]
                elapsed=time.perf_counter()-before
                self.assertLess(elapsed,.25)
                self.assertFalse(point["available"])
                self.assertEqual(point["weatherSessionModel"]["fitStatus"]["state"],"fitting")
                self.assertTrue(worker.is_alive())
            finally:
                release.set()
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0]["state"],"ready")
        with patch.object(forecast,"_trained_model",side_effect=AssertionError("daily artifact must remain pinned")):
            self.assertEqual(forecast.refresh_model("unused",ISSUE)["state"],"ready")


if __name__=="__main__":
    unittest.main()
