"""Exercise live arrival integration with isolated SQLite and cached learners.

The dashboard computation, arrival inputs, probability inference and publication
seams remain real. Unrelated heavy models and provider paths are stubbed.
"""

from pathlib import Path as _PublicPath
import sys as _public_sys
_public_sys.path.insert(0, str(_PublicPath(__file__).resolve().parents[1] / "app"))
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import BukitKiara_Dashboard as app
import fresh_sensor_features
from forecast_clock import ForecastClock
from forecast_payload import loads


# Follow the production alias so the test exercises the actually selected recipe.
arrival_change_model = app.arrival_change_model


class CachedClassifier:
    classes_ = np.arange(5)

    def __init__(self):
        self.queries = []

    def predict_proba(self, matrix):
        self.queries.append(np.asarray(matrix).copy())
        return np.tile([.01, .14, .70, .10, .05], (len(matrix), 1))


class ArrivalDashboardIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.issue = 1791450145
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.database = Path(self.folder.name) / "dashboard.db"
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(app, "DB_PATH", self.database))
        self.stack.enter_context(patch.object(app.time, "time", return_value=self.issue))
        self.stack.enter_context(patch.object(app.time, "sleep"))
        app.init_db()
        self.rows = [{
            "epoch": self.issue - 30 - 180 * (60 - index),
            "timestamp": app.iso_from_epoch(self.issue - 30 - 180 * (60 - index)),
            "pm02": 140 - .6 * index, "pm10": 145 - .6 * index,
            "atmp": 29.4, "rhum": 55.0, "heatindex": 36.0,
        } for index in range(61)]
        with app.db() as connection:
            connection.executemany(
                "INSERT INTO readings(epoch,timestamp,pm02,pm10,atmp,rhum,heatindex) "
                "VALUES(:epoch,:timestamp,:pm02,:pm10,:atmp,:rhum,:heatindex)", self.rows,
            )
            # A future observation must not enter this issue's model inputs.
            connection.execute(
                "INSERT INTO readings(epoch,timestamp,pm02,pm10,atmp,rhum,heatindex) VALUES(?,?,?,?,?,?,?)",
                (self.issue + 10, "future", 99999.0, 99999.0, 99.0, 99.0, 99.0),
            )
        self.reference = fresh_sensor_features.feature_metadata(self.rows, self.issue)["freshReferencePm25"]
        self.classifier = CachedClassifier()
        cutoff = arrival_change_model.midnight_epoch(self.issue)
        self.artifact = {
            "columns": list(getattr(arrival_change_model, "FEATURE_COLUMNS", app.arrival_feature_inputs.FEATURE_COLUMNS)),
            "modelVersion": arrival_change_model.MODEL_VERSION,
            "adapterVersion": arrival_change_model.ADAPTER_VERSION,
            "trainingCutoffEpoch": cutoff, "identity": arrival_change_model._identity(),
            "estimator": self.classifier,
            "metadata": {"available": True, "fittedAtEpoch": self.issue - 60,
                         "trainingCutoffEpoch": cutoff, "maxTrainingOutcomeEpoch": cutoff - 1},
        }
        self.stack.enter_context(patch.object(arrival_change_model, "_load_artifact", return_value=self.artifact))
        self.predict_spy = self.stack.enter_context(patch.object(
            arrival_change_model, "predict_arrival", wraps=arrival_change_model.predict_arrival,
        ))
        self.input_spy = self.stack.enter_context(patch.object(
            app.arrival_feature_inputs, "load_inputs", wraps=app.arrival_feature_inputs.load_inputs,
        ))
        self.refresh = self.stack.enter_context(patch.object(
            arrival_change_model, "refresh_model", side_effect=AssertionError("Inference must not fit"),
        ))
        self.fit = self.stack.enter_context(patch.object(
            arrival_change_model, "fit_prepared", side_effect=AssertionError("Inference must not fit"),
        ))
        self.stack.enter_context(patch.object(
            app, "qualify_session_outputs", side_effect=AssertionError("Manual selector revived"),
        ))
        self.parent = {
            "available": True, "state": "rapid_improvement",
            "label": "Rapid PM2.5 reduction detected", "analysisBucketEndEpoch": self.rows[-1]["epoch"],
            "forecastClock": ForecastClock(self.issue, self.issue // 900 * 900).metadata(),
            "arrival": {"available": False, "pointRole": "raw_model_output"},
            "trail": {"available": False, "pointRole": "raw_model_output"},
        }
        self.parent_spy = self.stack.enter_context(patch.object(
            app, "air_window_analysis", side_effect=lambda *args: deepcopy(self.parent),
        ))
        self.numeric = {
            "available": True, "modelVersion": app.fresh_numeric_model.MODEL_VERSION,
            "forecastClock": app.fresh_numeric_model.forecast_clock(self.issue),
            "arrivalPoint": 111.12345678901234, "trailMeanPoint": 109.98765432101234,
            "trailMinimumPoint": 91.1, "trailMaximumPoint": 129.2,
            "freshReferencePm25": self.reference,
            "inputLineage": fresh_sensor_features.feature_metadata(self.rows, self.issue),
        }
        self.numeric_spy = self.stack.enter_context(patch.object(
            app.fresh_numeric_model, "predict", side_effect=lambda *args: deepcopy(self.numeric),
        ))
        for name, value in {
            "passing_shower_signal": {"state": "none"},
            "rapid_clearance_event_outlook": {"available": False},
            "shadow_dry_dispersion_watch": {"available": False},
            "weather_outlook": {"available": False, "windows": {"morning": {}, "afternoon": {}}},
            "latest_weather_payload": None,
            "latest_air_quality_payload": None,
            "regional_haze_window_outlook": {"available": False},
            "paired_history": ({"morning": {}, "afternoon": {}}, [], None),
            "coach_api_contract_metadata": {},
        }.items():
            self.stack.enter_context(patch.object(app, name, return_value=value))
        self.stack.enter_context(patch.object(
            app, "attach_experimental_rapid_change_risk",
            side_effect=lambda air, *args: {**air, "experimentalRapidChangeRisk": {"available": False}},
        ))
        self.stack.enter_context(patch.object(
            app, "attach_experimental_short_horizon_change", side_effect=lambda air, *args: air,
        ))
        self.stack.enter_context(patch.object(
            app, "attach_first_crossing_event_forecast",
            side_effect=lambda air, *args: {**air, "firstCrossingEventForecast": {"available": False}},
        ))
        self.stack.enter_context(patch.object(
            app, "attach_cycling_window_forecast",
            side_effect=lambda air, *args: {**air, "cyclingWindowForecast": {"available": False}},
        ))
        self.stack.enter_context(patch.object(app.regional_pm_inputs, "describe_context", return_value={"available": False}))
        self.morning_spy = self.stack.enter_context(patch.object(
            app.weather_session_forecast, "predict_windows", return_value={"morning": {"available": False}},
        ))
        self.stack.enter_context(patch.object(app.patchtst_session_forecast, "read_issued_sources", return_value={"weather": [], "cams": []}))
        self.stack.enter_context(patch.object(app.patchtst_session_forecast, "materialize_snapshot", return_value={}))
        self.stack.enter_context(patch.object(
            app.patchtst_session_forecast, "apply_experimental_afternoon",
            side_effect=lambda rows, windows, issue, estimates, **kwargs: {**estimates, "afternoon": {"available": False}},
        ))

    def compute(self):
        return app._compute_analysis(28, include_trials=False)

    def test_live_computation_reaches_cached_classifier_with_causal_rows_and_shared_reference(self):
        result = self.compute()
        self.predict_spy.assert_called_once()
        call = self.predict_spy.call_args
        self.assertEqual(call.args[2], self.issue)
        self.assertEqual(call.args[3], self.rows[-1]["epoch"])
        self.assertEqual(call.args[4], self.reference)
        self.assertTrue(all(row["epoch"] <= self.issue for row in call.args[1]))
        self.assertEqual(self.input_spy.call_args.kwargs["sensor_rows"], call.args[1])
        distribution = result["airWindow"]["arrivalChangeForecast"]
        self.assertTrue(distribution["available"], distribution)
        self.assertEqual(distribution["modelVersion"], arrival_change_model.MODEL_VERSION)
        self.assertEqual(distribution["featureColumns"], self.artifact["columns"])
        self.assertEqual(distribution["modelIdentity"], self.artifact["identity"])
        self.assertEqual(distribution["forecastIssuedEpoch"], self.issue)
        self.assertEqual(distribution["arrivalEpoch"], self.issue + 5400)
        self.assertEqual(distribution["referencePm"], result["airWindow"]["arrival"]["baselinePoint"])
        self.assertEqual(distribution["classProbabilities"], [.01, .14, .70, .10, .05])
        self.assertEqual(distribution["featureSourceMaxEpoch"], self.rows[-1]["epoch"])
        self.assertLessEqual(distribution["featureProvenance"]["forecastIssuedEpoch"], self.issue)
        self.assertEqual(len(self.classifier.queries), 1)
        self.assertTrue(np.isfinite(self.classifier.queries[0][0, 0]))
        self.refresh.assert_not_called()
        self.fit.assert_not_called()
        self.assertFalse(self.morning_spy.call_args.kwargs["allow_fit"])

    def test_probability_survives_collecting_parent_and_missing_numeric_asset_in_all_consumers(self):
        self.parent = {"available": False, "state": "collecting", "arrival": {"available": False}, "trail": {"available": False}}
        self.numeric.update(available=False, reason="fresh_numeric_model_asset_unavailable", arrivalPoint=None,
                            trailMeanPoint=None, trailMinimumPoint=None, trailMaximumPoint=None)
        result = self.compute()
        air = result["airWindow"]
        self.assertTrue(air["arrivalChangeForecast"]["available"])
        self.assertFalse(air["arrival"]["available"])
        self.assertIsNone(air["arrival"]["point"])
        self.assertEqual(air["forecastClock"]["forecastIssuedEpoch"], self.issue)
        self.assertEqual(air["forecastClock"]["arrivalTargetEpoch"], self.issue + 5400)
        device = app.rlcd_api.build_payload(self.rows[-1], result, {}, self.issue + 20)
        self.assertFalse(device["forecast"]["near90"]["available"])
        self.assertIsNone(device["forecast"]["near90"]["pm25_ugm3"])
        self.assertTrue(device["forecast"]["near90"]["arrival_change"]["available"])
        with patch.object(app, "analysis", return_value=result):
            coach = app.ride_conditions_api(now_epoch=self.issue + 20)
        self.assertTrue(coach["exposureOutlook"]["arrivalChangeForecast"]["available"])
        self.assertIsNone(coach["exposureOutlook"]["arrival"]["projectedPm25UgM3"])
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node is required for actual frontend integration")
        frontend = '''
const fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(SOURCE_PATH,'utf8');
const html=source.match(/HTML = r"""([\\s\\S]*?)"""/)[1];
const script=html.match(/<script>([\\s\\S]*?)<\\/script>/)[1];
const nodes=new Map();
const node=id=>{if(!nodes.has(id))nodes.set(id,{textContent:'',hidden:false,dataset:{},style:{},classList:{add(){},remove(){}}});return nodes.get(id);};
class FrozenDate extends Date { static now(){return FROZEN_NOW;} }
vm.runInNewContext(script.slice(0,script.indexOf('document.querySelectorAll("#ranges button")'))+'\\nrenderAnalysis(fixture);',
  {console,Date:FrozenDate,Math,Number,String,Boolean,Array,Object,Promise,document:{getElementById:node},fixture:FIXTURE});
console.log(JSON.stringify({visible:!node('arrivalChangeProbabilities').hidden,fall20:node('arrivalFall20Probability').textContent,fall40:node('arrivalFall40Probability').textContent,rise20:node('arrivalRise20Probability').textContent,rise40:node('arrivalRise40Probability').textContent,numeric:node('arrivalValue').textContent}));
'''
        frontend = frontend.replace("SOURCE_PATH", json.dumps(str(Path(app.__file__)))).replace("FROZEN_NOW", str((self.issue + 20) * 1000)).replace("FIXTURE", json.dumps(result))
        completed = subprocess.run([node, "-"], input=frontend, text=True, encoding="utf-8", capture_output=True, check=True)
        rendered = json.loads(completed.stdout)
        self.assertTrue(rendered["visible"])
        self.assertEqual([rendered[key] for key in ("fall20", "fall40", "rise20", "rise40")], ["15%", "1%", "15%", "5%"])
        self.assertEqual(rendered["numeric"], "—")

    def test_actual_publication_archives_available_distribution_and_only_fresh_issue(self):
        result = self.compute()
        result["computation"] = {}
        stop = Mock()
        stop.is_set.side_effect = [False, True]
        output = Path(self.folder.name) / "issued.jsonl"
        original_writer = arrival_change_model.record_issued_prediction
        def write_prediction(value, **kwargs):
            return original_writer(value, output_path=output, **kwargs)
        with ExitStack() as publication:
            publication.enter_context(patch.object(app, "_compute_analysis", return_value=result))
            for method, value in (("schedule", None), ("take_job", 28), ("publish_primary", None), ("finish", None)):
                publication.enter_context(patch.object(app.analysis_delivery, method, return_value=value))
            for module in (app.window_pm_predictor, app.weather_session_forecast, app.afternoon_xgboost_forecast):
                publication.enter_context(patch.object(module, "record_issue"))
            publication.enter_context(patch.object(app.fresh_event_model, "record_issued_prediction"))
            publication.enter_context(patch.object(app.cycling_window_live, "record_issued_prediction"))
            writer = publication.enter_context(patch.object(arrival_change_model, "record_issued_prediction", side_effect=write_prediction))
            app.analysis_worker(stop)
        writer.assert_called_once()
        record = json.loads(output.read_text(encoding="utf-8").strip())
        self.assertTrue(record["available"])
        self.assertTrue(record["isOriginallyIssued"])
        self.assertEqual(record["forecastIssuedEpoch"], self.issue)
        self.assertEqual(record["classProbabilities"], [.01, .14, .70, .10, .05])
        self.assertEqual(record["inputSnapshotId"], result["inputProvenance"]["snapshotId"])
        with app.db() as connection:
            saved = loads(connection.execute("SELECT payload FROM dashboard_forecast_issues").fetchone()[0])
        self.assertEqual(saved["airWindow"]["arrivalChangeForecast"]["classProbabilities"], [.01, .14, .70, .10, .05])
        self.assertEqual(saved["airWindow"]["arrivalChangeForecast"]["forecastIssuedEpoch"], self.issue)
        with self.assertRaises(ValueError):
            original_writer(result["airWindow"]["arrivalChangeForecast"], output_path=output, recorded_epoch=self.issue + 121)
        self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

    def test_analysis_http_request_serves_cached_class_without_computation_or_fitting(self):
        result = self.compute()
        handler = object.__new__(app.Handler)
        handler.path = "/api/analysis?days=28"
        captured = []
        handler.send_json = lambda value, *args: captured.append(value)
        with patch.object(app.analysis_delivery, "response", return_value=result), \
                patch.object(app, "_compute_analysis", side_effect=AssertionError("HTTP must not compute")):
            handler.do_GET()
        self.assertTrue(captured[0]["airWindow"]["arrivalChangeForecast"]["available"])
        self.assertEqual(len(self.classifier.queries), 1)
        self.refresh.assert_not_called()
        self.fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
