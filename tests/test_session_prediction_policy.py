"""Meaningful contract tests; no live database, fitting, network or artifacts."""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
import tempfile
from pathlib import Path
import sqlite3
import json
from contextlib import closing
from unittest.mock import patch

import numpy as np
import pandas as pd
import session_prediction_policy as policy
import patchtst_session_forecast as patchtst
import afternoon_xgboost_forecast as xgb

TZ=timezone(timedelta(hours=8))
ISSUE=int(datetime(2026,10,8,12,tzinfo=TZ).timestamp())
START=ISSUE+7200
END=START+7200


def value(raw=119.9):
    return {"available":True,"modelVersion":"candidate_v2","sensorAnchor":100.,
            "sensorReferenceEpoch":ISSUE-30,"sensorReferenceCount":1,
            "mean":raw,"prediction":raw,"forecastedAtEpoch":ISSUE,
            "startEpoch":START,"endEpoch":END,
            "candidateForecast":{"available":True,"modelVersion":"candidate_v2",
                                 "mean":raw,"numericalPolicyIdentifier":"raw_recipe_v2",
                                 "forecastedAtEpoch":ISSUE,"startEpoch":START,"endEpoch":END}}


class PolicyTests(unittest.TestCase):
    def test_unqualified_raw_is_visible_but_operational_point_is_baseline(self):
        original=value()
        result=policy.apply_session_prediction_policy(original)
        self.assertEqual(result["mean"],100.)
        self.assertEqual(result["candidateForecast"]["mean"],119.9)
        self.assertEqual(result["modelSelection"]["selectedPolicy"],"same_issue_persistence")
        self.assertFalse(result["modelSelection"]["skillGatePassed"])
        self.assertEqual(original,value())

    def test_development_or_wrong_version_evidence_cannot_promote(self):
        evidence={"count":1000,"distinctDays":30,"mae":1.,"persistenceMae":10.,
                  "skillGatePassed":True,"validationMode":"historical_replay"}
        result=policy.apply_session_prediction_policy(value(),evidence=evidence)
        self.assertEqual(result["mean"],100.)
        self.assertFalse(result["qualificationPolicy"]["qualified"])

    def test_genuine_matched_sufficient_issued_evidence_can_promote(self):
        evidence={"asIssuedBeforeOutcome":True,"independentOrigins":True,
                  "incompleteTargetsScored":False,"candidateModelVersion":"candidate_v2",
                  "numericalPolicyIdentifier":"raw_recipe_v2","scope":"same_day_afternoon",
                  "targetClockMatched":True,"issueClockMatched":True,
                  "matchedIssueLocalSeconds":12*3600,"matchedTargetStartLocalSeconds":14*3600,
                  "matchedDurationSeconds":7200,"matchedActualLeadSeconds":7200,
                  "issueClockToleranceSeconds":0,
                  "validationMode":"as_issued_same_session_target_lead_and_local_issue_clock",
                  "count":14,"distinctDays":14,"mae":10.,"persistenceMae":15.,
                  "rmse":12.,"persistenceRmse":17.,"p90AbsoluteError":19.,
                  "persistenceP90AbsoluteError":25.,"positiveDayFraction":.8}
        result=policy.apply_session_prediction_policy(value(),evidence=evidence)
        self.assertEqual(result["mean"],119.9)
        self.assertTrue(result["qualificationPolicy"]["qualified"])
        for field,new in (("candidateModelVersion","candidate_v1"),("distinctDays",13),
                          ("issueClockMatched",False),("positiveDayFraction",.5),
                          ("mae",-1),("rmse",-1),("p90AbsoluteError",-1),
                          ("positiveDayFraction",5),("matchedActualLeadSeconds",6600),
                          ("matchedTargetStartLocalSeconds",14*3600+900),
                          ("count",14.5),("distinctDays",14.5),("mae",True)):
            wrong={**evidence,field:new}
            self.assertEqual(policy.apply_session_prediction_policy(value(),evidence=wrong)["mean"],100.)
        shifted=value()
        shifted["forecastedAtEpoch"]+=31
        shifted["candidateForecast"]["forecastedAtEpoch"]+=31
        self.assertEqual(policy.apply_session_prediction_policy(shifted,evidence=evidence)["mean"],100.)

    def test_candidate_clock_mismatch_cannot_leak_into_current_card(self):
        wrong=value()
        wrong["candidateForecast"]["startEpoch"]+=900
        result=policy.apply_session_prediction_policy(wrong)
        self.assertFalse(result["candidateForecast"]["available"])
        self.assertIsNone(result["candidateForecast"]["mean"])
        self.assertEqual(result["mean"],100.)

    def test_stale_future_or_unknown_operational_reference_is_unavailable(self):
        for epoch in (ISSUE-86400,ISSUE+60,None):
            wrong={**value(),"sensorReferenceEpoch":epoch}
            self.assertFalse(policy.apply_session_prediction_policy(wrong)["available"])

    def test_actual_lead_guard_is_not_closed_origin_lead(self):
        issue=START-105*60+1
        self.assertFalse(policy.candidate_scope(issue,START,END)["trainedLead"])
        self.assertTrue(policy.candidate_scope(issue-1,START,END)["trainedLead"])
        rows=[{"epoch":issue-60,"pm02":100.}]
        with patch.object(xgb,"ensure_background_fit",side_effect=AssertionError("must not fit invalid domain")):
            result=xgb.predict_windows("unused",rows,{"afternoon":{"startEpoch":START,"endEpoch":END}},issue)
        self.assertTrue(result["afternoon"]["available"])
        self.assertEqual(result["afternoon"]["startEpoch"],START)
        self.assertEqual(result["afternoon"]["mean"],100.)
        self.assertFalse(result["afternoon"]["candidateForecast"]["available"])

    def test_moving_target_never_uses_xgb_extrapolation(self):
        start=START+900
        rows=[{"epoch":ISSUE-30,"pm02":100.}]
        with patch.object(xgb,"ensure_background_fit",side_effect=AssertionError("must not fit moving target")):
            result=xgb.predict_windows("unused",rows,{"afternoon":{"startEpoch":start,"endEpoch":start+7200}},ISSUE)["afternoon"]
        self.assertTrue(result["available"])
        self.assertEqual(result["startEpoch"],start)
        self.assertEqual(result["candidateForecast"]["reason"],"outside_canonical_exact_session_target")

    def test_fresh_baseline_does_not_use_future_or_stale_sensor(self):
        rows=[{"epoch":ISSUE-60,"pm02":100.},{"epoch":ISSUE+1,"pm02":999.}]
        self.assertEqual(policy.baseline_session_result(rows,ISSUE,START,END)["mean"],100.)
        rows=[{"epoch":ISSUE-241,"pm02":100.}]
        self.assertFalse(policy.baseline_session_result(rows,ISSUE,START,END)["available"])

    def test_arbitrary_issue_clock_is_not_claimed_as_evaluated(self):
        scope=policy.candidate_scope(ISSUE+31,START,END)
        self.assertTrue(scope["rawCandidatePermitted"])
        self.assertFalse(scope["queryMatchesEvaluatedClock"])
        self.assertFalse(scope["movingWindowSkillVerified"])

    def test_xgb_historical_reference_excludes_known_late_sensor_revision(self):
        source={"rawEpochs":np.array([ISSUE-200,ISSUE-100,ISSUE-30]),
                "rawPm":np.array([100.,999.,102.]),
                "rawAvailableEpochs":np.array([ISSUE-150,ISSUE+100,ISSUE-10])}
        point,stamp,count=xgb._reference_at_issue(source,ISSUE,ISSUE-30)
        self.assertEqual(point,101.)
        self.assertEqual(stamp,ISSUE-30)
        self.assertEqual(count,2)

    def test_xgb_closed_features_with_late_receipts_are_withheld(self):
        origins=pd.to_datetime([ISSUE,ISSUE+900,ISSUE+1800],unit="s",utc=True)
        source={"sourceAvailableMax":pd.Series([ISSUE+100,ISSUE+800,np.nan],index=origins)}
        eligible=xgb._feature_receipts_available(source,origins,np.array([ISSUE,ISSUE+900,ISSUE+1800]))
        self.assertEqual(eligible.tolist(),[False,True,True])

    def test_fixed_patch_failure_does_not_choose_xgb_or_persistence(self):
        prior=value()
        prior["candidateForecast"]["modelVersion"]=xgb.MODEL_VERSION
        prior=policy.apply_session_prediction_policy(prior)
        fallback=patchtst.direct.unavailable_model_output(ISSUE,START,END,patchtst.MODEL_VERSION,
                                                         "daily_artifact_not_ready","afternoon")
        with patch.object(patchtst,"predict_windows",return_value={"afternoon":fallback}):
            result=patchtst.apply_experimental_afternoon([], {"afternoon":{"startEpoch":START,"endEpoch":END}},
                                                        ISSUE,{"afternoon":prior},snapshot={})["afternoon"]
        self.assertFalse(result["available"])
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["candidateForecast"]["mean"])
        self.assertEqual(result["candidateForecast"]["modelVersion"],patchtst.MODEL_VERSION)
        self.assertEqual(result["modelSelection"],{})
        self.assertEqual(result["reason"],"daily_artifact_not_ready")

    def test_xgb_archives_raw_candidate_separately_from_operational_baseline(self):
        issued=value()
        issued["candidateForecast"].update(modelVersion=xgb.MODEL_VERSION,
                                          numericalPolicyIdentifier=xgb.NUMERICAL_POLICY)
        issued.update(sensorWatermarkEpoch=ISSUE-30,featureSourceMaxEpoch=ISSUE-45,
                      inputAvailabilityPolicy=xgb.LIVE_INPUT_POLICY,inputSnapshotReceivedEpoch=ISSUE-5,
                      originEpoch=ISSUE//900*900,weatherFetchedEpoch=ISSUE-60,
                      weatherSessionModel={"candidateComputed":True,"applied":False,"featureVersion":xgb.FEATURE_VERSION,
                                           "trainingCutoffEpoch":xgb.training_cutoff(ISSUE),
                                           "trainingLatestTargetEndEpoch":xgb.training_cutoff(ISSUE)-900,
                                           "fittedAtEpoch":ISSUE-120})
        issued=policy.apply_session_prediction_policy(issued)
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/"test.db"
            self.assertEqual(xgb.record_issue(db,{"afternoon":issued},ISSUE+10),1)
            self.assertEqual(xgb.record_issue(db,{"afternoon":issued},ISSUE+20),0)
            with closing(sqlite3.connect(db)) as conn:
                version,raw,anchor,payload=conn.execute("SELECT model_version,predicted_mean,anchor_pm25,payload FROM window_pm_forecast_issues").fetchone()
            saved=json.loads(payload)
            self.assertEqual(version,xgb.MODEL_VERSION)
            self.assertEqual(raw,119.9)
            self.assertEqual(anchor,100.)
            self.assertEqual(saved["operationalMean"],100.)
            self.assertEqual(saved["publishedModelVersion"],policy.BASELINE_MODEL_VERSION)
            self.assertEqual(saved["archiveRole"],"raw_candidate_before_operational_qualification")


class PatchWrapperTests(unittest.TestCase):
    def test_source_adapter_preserves_latest_available_vintage_and_receipts(self):
        receipt={"receiptKnown":True,"availableEpoch":ISSUE-10,"revisionId":"available-old-revision"}
        weather=[{"fetched_epoch":ISSUE-60,"source":"weather","payload":"{}","_provenance":receipt}]
        cams=[{"fetched_epoch":ISSUE-60,"source":"Open-Meteo / CAMS Global","model_version":"cams_anchor_v2",
               "payload":"{}","_provenance":receipt},
              {"fetched_epoch":ISSUE-30,"source":"other","model_version":"cams_anchor_v2",
               "payload":"{}","_provenance":{"receiptKnown":False}}]
        def runs(conn,table,cutoff,**kw):
            self.assertEqual(cutoff,ISSUE)
            self.assertEqual(kw["start_fetched_epoch"],ISSUE-7200)
            return weather if table=="weather_forecast_runs" else cams
        with patch.object(patchtst.provenance,"compatible_source_runs",side_effect=runs):
            sources=patchtst.read_issued_sources(object(),ISSUE)
        self.assertEqual(sources["weather"],[(ISSUE-60,"weather","{}")])
        self.assertEqual(sources["cams"],[(ISSUE-60,"Open-Meteo / CAMS Global","{}")])
        self.assertEqual(sources["sourceProvenance"]["weather"][0]["revisionId"],"available-old-revision")

    def run_prediction(self,raw):
        mark=ISSUE-60
        bundle={"past":np.zeros((1,96,3)),"future":np.zeros((1,26,15)),
                "context":np.zeros((1,31)),"fresh":np.array([100.])}
        audit={"fresh":100.,"weatherFetchedEpoch":ISSUE-30,"sensorReferenceEpoch":mark,
               "sequenceSourceMaxEpoch":mark,"pastPmValidSteps":96,"tensorShapes":{},
               "contextFeatures":{"camsTarget":float("nan")},"closedLevel":99.,
               "sensorReferenceCount":1,"sensorWatermarkEpoch":mark,"featureSourceMaxEpoch":mark,
               "camsFetchedEpoch":None,"weatherPayloadSha256":"weather-hash"}
        adapter=SimpleNamespace(query=lambda snapshot,**kw:(bundle,audit),safe=lambda v:v,
                                prepare=SimpleNamespace(VERSION="original_adapter",
                                    Sources=lambda snapshot:SimpleNamespace(archive=None),
                                    rain=SimpleNamespace(describe_window=lambda *args:{})))
        cutoff=patchtst.training_cutoff(ISSUE)
        fitted={"fittedAtEpoch":ISSUE-100,"trainingRows":120,"distinctTrainingTargetDates":10,
                "maximumTrainingCompleteEpoch":cutoff-900,"trainingTensorLabelSha256":"training-hash",
                "checkpointSha256":"unchanged-original-checkpoint","dailyPolicySha256":"daily-policy"}
        ready={"cutoff":cutoff,"fitted":fitted,"adapter":adapter,"metadataSha256":"metadata",
               "policy":{"liveRuntimeRevision":"synthetic-test-receipt-inputs",
                         "liveRuntimeRevisionSha256":"synthetic-test-revision-sha"},
               "model":SimpleNamespace(predict=lambda *args:np.array([raw]))}
        snapshot={"issueEpoch":ISSUE,"cutoffEpoch":ISSUE,"sensorWatermarkEpoch":mark,
                  "dbSnapshotOpenedEpoch":ISSUE-10,"inputSnapshotReceivedEpoch":ISSUE-8,
                  "readOnly":True,"futureObservedSensorRowsLoaded":0,
                  "sensor":[{"epoch":mark,"pm02":100.}],
                  "weather":[(ISSUE-30,"issued-weather",{})],"cams":[]}
        with patch.object(patchtst,"_READY",ready),patch.object(patchtst.time,"time",return_value=ISSUE+2), \
             patch.object(patchtst.direct,"patch_query",return_value=(bundle,audit)):
            return patchtst.predict_windows(snapshot["sensor"],{"afternoon":{"startEpoch":START,"endEpoch":END}},
                                            ISSUE,snapshot=snapshot,weather_fetched_epoch=ISSUE-30)["afternoon"]

    def test_twenty_event_threshold_does_not_discontinuously_replace_raw_estimate(self):
        left,right=self.run_prediction(119.9),self.run_prediction(120.1)
        self.assertTrue(left["available"])
        self.assertAlmostEqual(right["candidateForecast"]["mean"]-left["candidateForecast"]["mean"],.2)
        self.assertEqual(left["mean"],119.9)
        self.assertEqual(right["mean"],120.1)
        self.assertEqual(left["candidateForecast"]["mean"],left["mean"])
        self.assertEqual(left["pointRole"],"experimental_model_output")
        self.assertFalse(left["modelOutputPolicy"]["numericalSelectionApplied"])
        self.assertFalse(left["patchtstModel"]["deadbandApplied"])
        self.assertEqual(left["patchtstModel"]["eventThresholdUgM3"],20)
        self.assertEqual(left["patchtstModel"]["checkpointSha256"],"unchanged-original-checkpoint")
        self.assertEqual(left["patchtstModel"]["legacyModelVersion"],patchtst.LEGACY_MODEL_VERSION)


if __name__=="__main__":
    unittest.main()
