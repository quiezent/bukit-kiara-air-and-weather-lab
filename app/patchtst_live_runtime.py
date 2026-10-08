"""Strict live-input revision of the unchanged original Patch daily runtime.

The original forward/daily freezes continue to reject changed dependencies.
This separately frozen inference wrapper permits exactly three reviewed input
readers, verifies preserved original bytes, and never trains or edits an old
policy. A further input change requires a new declared revision.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import time
from contextlib import closing
import sqlite3
import urllib.request

VERSION = "patchtst_live_receipt_inputs_revision_v2"
ROOT = Path(__file__).resolve().parent
REVISION_DIR = ROOT / "research/patch_dashboard_deployment_20261008/live_receipt_revision_v2"
REVISED_INPUTS = frozenset(("afternoon_direction_features.py", "rain_weather_features.py", "window_pm_predictor.py"))
ADDITIONAL_INPUTS = frozenset(("observation_provenance.py",))
_LOCK = threading.RLock()
_RUNTIME = None
DAILY_REVISIONS = (
    ("DAILY_RUNTIME_REVISION", None),
    ("DAILY_PAIRING_REVISION", "daily_cache_runtime_v1.py"),
    ("DAILY_SOURCE_GUARD_REVISION", "daily_cache_pairing_v2.py"),
    ("DAILY_AFTERNOON_BLEND_REVISION", "daily_cache_source_guard_v3.py"),
    ("DAILY_AFTERNOON_RAW_RETRY_REVISION", "daily_cache_afternoon_v4.py"),
    ("DAILY_WIND61_SUPERSESSION_REVISION", "daily_cache_afternoon_retry_v5.py"),
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_runtime(cache, *, revision_dir=None):
    """Verify the exact amended inference dependency chain; no generic waiver."""
    directory = REVISION_DIR if revision_dir is None else Path(revision_dir)
    declaration_path, frozen_path = directory / "REVISION.json", directory / "REVISION_FREEZE.json"
    declared, sealed = _json(declaration_path), _json(frozen_path)
    if (sealed.get("version") != VERSION or declared.get("version") != VERSION
            or sealed.get("revisionSha256") != _sha(declaration_path)
            or sealed.get("verifierSha256") != _sha(Path(__file__))):
        raise RuntimeError("Declared live receipt runtime revision differs")
    predecessor = ROOT / "research/patch_dashboard_deployment_20261008/live_receipt_revision_v1"
    if (declared.get("previousRevisionSha256") != _sha(predecessor / "REVISION.json")
            or declared.get("previousRevisionFreezeSha256") != _sha(predecessor / "REVISION_FREEZE.json")
            or declared.get("previousVerifierSha256") != _sha(predecessor / "verifier_source.py")):
        raise RuntimeError("Preserved predeployment inference revision changed")
    if (set(declared.get("revisedInputs", {})) != REVISED_INPUTS
            or set(declared.get("additionalInputs", {})) != ADDITIONAL_INPUTS):
        raise RuntimeError("Only the three declared receipt-reader revisions are permitted")
    if Path(cache.HERE).resolve() != (ROOT / "research/sequence_followup_20261008/forward").resolve():
        raise RuntimeError("Unexpected original daily runtime location")
    here, original = Path(cache.HERE), cache.original
    if (Path(original.__file__).resolve() != (here / "forward_patchtst.py").resolve()
            or Path(cache.__file__).resolve() != (here / "daily_cache.py").resolve()):
        raise RuntimeError("Unexpected original runtime adapter")
    for filename, digest in declared["originalPolicyHashes"].items():
        if _sha(here / filename) != digest:
            raise RuntimeError("Preserved original policy/source changed: " + filename)
    if set(declared["originalPolicyHashes"]) != {
            "FREEZE.json", "POLICY.json", "DAILY_FREEZE.json", "DAILY_POLICY.json", "daily_cache.py", "forward_patchtst.py"}:
        raise RuntimeError("Incomplete original policy/source declaration")

    forward = _json(here / "FREEZE.json")
    if forward["policySha256"] != _sha(here / "POLICY.json"):
        raise RuntimeError("Original forward policy changed")
    for relative, digest in forward["sourceSha256"].items():
        if relative in REVISED_INPUTS:
            changed = declared["revisedInputs"][relative]
            preserved = directory / "original_inputs" / relative
            if (changed.get("originalSha256") != digest or _sha(preserved) != digest
                    or _sha(ROOT / relative) != changed.get("revisedSha256")):
                raise RuntimeError("Pinned live input revision changed: " + relative)
        elif _sha(ROOT / relative) != digest:
            raise RuntimeError("Unamended original source changed: " + relative)
    for relative, digest in declared["additionalInputs"].items():
        if _sha(ROOT / relative) != digest:
            raise RuntimeError("Pinned receipt prerequisite changed: " + relative)

    # The original daily runtime source has a pre-existing explicit revision
    # chain. Verify that same chain and every preserved source, without calling
    # or weakening the old verifier that correctly rejects the new inputs.
    frozen = _json(here / "DAILY_FREEZE.json")
    if (frozen["originalFreezeSha256"] != _sha(here / "FREEZE.json")
            or frozen["prefixArraySha256"] != _sha(original.DATA / "session_train.npz")
            or frozen["prefixMetadataSha256"] != _sha(original.DATA / "session_train_metadata.pkl")):
        raise RuntimeError("Original certified training prefix changed")
    expected, previous = frozen["dailyCacheSourceSha256"], None
    for stem, preserved in DAILY_REVISIONS:
        revision_path, freeze_path = here / (stem + ".json"), here / (stem + "_FREEZE.json")
        if not revision_path.exists():
            continue
        revision, freeze = _json(revision_path), _json(freeze_path)
        if (freeze["revisionSha256"] != _sha(revision_path)
                or revision["sourceSha256Before"] != expected
                or (previous is not None and revision["previousRevisionFreezeSha256"] != _sha(previous))
                or (preserved is not None and _sha(here / preserved) != expected)):
            raise RuntimeError("Original declared daily runtime revision chain differs")
        if stem == "DAILY_RUNTIME_REVISION" and revision["originalDailyFreezeSha256"] != _sha(here / "DAILY_FREEZE.json"):
            raise RuntimeError("Original daily freeze changed")
        expected, previous = revision["sourceSha256After"], freeze_path
    if (frozen["policySha256"] != _sha(here / "DAILY_POLICY.json")
            or expected != _sha(here / "daily_cache.py")
            or frozen["originalAdapterSha256"] != _sha(here / "forward_patchtst.py")
            or frozen["modelSources"] != original.trainer.source_hashes()):
        raise RuntimeError("Original daily prospective source/policy changed")
    return {**_json(here / "DAILY_POLICY.json"), "liveRuntimeRevision": VERSION,
            "liveRuntimeRevisionSha256": sealed["revisionSha256"],
            "originalFrozenInputsPreserved": True,
            "inputRevisionInvalidatesPreviousIssuedSkill": True}


def wrap_cache(cache):
    """Select the explicitly revised isolated cache, never mutate the original."""
    if getattr(cache, "_live_receipt_runtime_version", None) == VERSION:
        verify_runtime(cache)
        return cache
    return get_cache(cache)


def _revision_tags(cache):
    policy = verify_runtime(cache)
    return {"liveRuntimeRevision": VERSION,
            "liveRuntimeRevisionSha256": policy["liveRuntimeRevisionSha256"],
            "originalDailyPolicySha256": _sha(cache.HERE / "DAILY_POLICY.json"),
            "inputRevisionInvalidatesPreviousIssuedSkill": True,
            "sourceReceiptPolicy": "immutable_available_revision_post_commit_visibility_legacy_unknown"}


def _clone_source(name, path, expected):
    code = Path(path).read_bytes()
    if hashlib.sha256(code).hexdigest() != expected:
        raise RuntimeError("Declared source changed before isolated load: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register only the new name. Imports and globals of original modules are
    # never assigned or patched by this bridge.
    sys.modules[name] = module
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


def _receipt_capture(cache, db_path, api="http://127.0.0.1:8765"):
    """Same original publication/watermark clocks with available ledger inputs."""
    import observation_provenance as provenance
    requested = time.time()
    with urllib.request.urlopen(api + "/api/analysis?days=28", timeout=30) as response:
        analysis = json.load(response)
    api_received = time.time()
    publication, mark = int(analysis["forecastIssuedEpoch"]), int(analysis["current"]["epoch"])
    if mark > publication or publication > api_received or api_received - publication > 180:
        raise RuntimeError("API publication is stale or has invalid source clocks")
    opened = time.time()
    path = Path(db_path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        records = provenance.compatible_sensor_rows(conn, opened,
            start_epoch=mark - 25 * 3600, end_epoch=mark)
        weather_rows = provenance.compatible_source_runs(conn, "weather_forecast_runs", opened,
            start_fetched_epoch=max(0, int(opened) - 7200))[-1:]
        cams_rows = [row for row in provenance.compatible_source_runs(conn, "air_quality_forecast_runs", opened,
            start_fetched_epoch=max(0, int(opened) - 7200))
            if row.get("model_version") in ("cams_anchor_v1", "cams_anchor_v2")
            and row.get("source") == "Open-Meteo / CAMS Global"][-1:]
    received = time.time()
    issue = math.ceil(received)
    if issue > time.time():
        time.sleep(issue - time.time())
    sensor = [{key: row.get(key) for key in ("epoch", "pm02", "atmp", "rhum")} for row in records]
    def sources(rows):
        return [(int(row["fetched_epoch"]), str(row["source"]), row["payload"]) for row in rows]
    weather, cams = sources(weather_rows), sources(cams_rows)
    if not sensor or int(sensor[-1]["epoch"]) != mark:
        raise RuntimeError("Available DB snapshot lacks the published sensor watermark")
    if not weather or not 0 <= issue - weather[0][0] <= 7200:
        raise RuntimeError("Original weather unavailable or stale for this query")
    if not 0 <= issue - mark <= 240:
        raise RuntimeError("Published reference is older than240seconds; recapture next publication")
    if not mark <= publication <= api_received <= received <= issue:
        raise RuntimeError("Invalid original source publication/receipt ordering")
    source_manifest = {"weather": [dict(row["_provenance"]) for row in weather_rows],
                       "cams": [dict(row["_provenance"]) for row in cams_rows],
                       "sensor": [dict(row.get("_provenance") or {}) for row in records]}
    return {"cutoffEpoch": issue, "sensor": sensor, "weather": sorted(weather), "cams": sorted(cams),
            "sensorWatermarkEpoch": mark, "apiRequestStartedEpoch": requested,
            "apiSnapshotReceivedEpoch": api_received, "apiPublishedForecastIssueEpoch": publication,
            "dbSnapshotOpenedEpoch": opened, "inputSnapshotReceivedEpoch": received,
            "issueEpoch": issue, "readOnly": True, "futureObservedSensorRowsLoaded": 0,
            "dashboardBuild": analysis.get("dashboardBuild"), "sourceProvenance": source_manifest,
            "incumbentAtOriginalApiIssue": {key: (analysis.get("windows") or {}).get(key) for key in ("morning", "afternoon")},
            "sourceSensorSha256": hashlib.sha256(json.dumps(sensor, sort_keys=True).encode()).hexdigest(),
            **_revision_tags(cache)}


def get_cache(original_cache=None):
    """Hash-verified original code in isolated modules and a separate cache."""
    global _RUNTIME
    if original_cache is None:
        forward = ROOT / "research/sequence_followup_20261008/forward"
        if str(forward) not in sys.path:
            sys.path.insert(0, str(forward))
        import daily_cache as original_cache
    verify_runtime(original_cache)
    with _LOCK:
        if _RUNTIME is not None:
            verify_runtime(_RUNTIME)
            return _RUNTIME
        declared = _json(REVISION_DIR / "REVISION.json")
        original = _clone_source("receipt_forward_patchtst_v2", original_cache.HERE / "forward_patchtst.py",
                                declared["originalPolicyHashes"]["forward_patchtst.py"])
        cache = _clone_source("receipt_daily_patchtst_v2", original_cache.HERE / "daily_cache.py",
                             declared["originalPolicyHashes"]["daily_cache.py"])
        cache.original = original
        cache.CACHE = cache.HERE / "receipt_runtime_v2_lifecycle"
        cache._live_receipt_runtime_version = VERSION
        cache.verify_policy = lambda: verify_runtime(cache)
        original.verify = lambda: {**_json(original.HERE / "POLICY.json"), **_revision_tags(cache)}
        original.capture = lambda db_path=ROOT / "bukit_kiara_air_history.db", api="http://127.0.0.1:8765": _receipt_capture(cache, db_path, api)
        original_write, atomic = original.write, cache._atomic
        original.write = lambda path, value: original_write(path, {**value, **_revision_tags(cache)})
        cache._atomic = lambda path, value: atomic(path, {**value, **_revision_tags(cache)})
        # Preserve compact receipt evidence in each stored input/audit row.
        original_query = original.query
        def query(snapshot, start=None, end=None):
            bundle, audit = original_query(snapshot, start, end)
            audit.update(_revision_tags(cache))
            audit["sourceReceiptManifestSha256"] = hashlib.sha256(json.dumps(
                snapshot.get("sourceProvenance", {}), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            audit["sourceReceiptKnownCounts"] = {key: sum(row.get("receiptKnown") is True for row in rows)
                                                 for key, rows in snapshot.get("sourceProvenance", {}).items()}
            audit["externalSourceReceiptEvidence"] = {key: rows for key, rows in
                snapshot.get("sourceProvenance", {}).items() if key in ("weather", "cams")}
            return bundle, audit
        original.query = query
        # Every artifact loaded through the new cache must have this explicit
        # input identity, even when its unchanged checkpoint was adopted.
        old_cached = cache._cached
        def cached(cutoff):
            metadata = _json(cache.CACHE / "models" / f"{cutoff}_patchtst.json")
            tags = _revision_tags(cache)
            if any(metadata.get(key) != tags[key] for key in ("liveRuntimeRevision", "liveRuntimeRevisionSha256", "originalDailyPolicySha256")):
                raise RuntimeError("Revised daily artifact lacks matching input policy identity")
            return old_cached(cutoff)
        cache._cached = cached
        _RUNTIME = cache
        return cache


def _adopt_completed_artifact(cache, cutoff, issue_epoch):
    """Copy a verified original completed checkpoint once, without any fit."""
    verify_runtime(cache)
    cutoff, issue_epoch = int(cutoff), int(issue_epoch)
    source = cache.HERE / "runtime/models" / f"{cutoff}_patchtst.pt"
    destination = cache.CACHE / "models" / source.name
    metadata_path = destination.with_suffix(".json")
    if destination.exists() and metadata_path.exists():
        metadata = _json(metadata_path)
        if (_sha(destination) != metadata.get("checkpointSha256")
                or any(metadata.get(key) != value for key, value in _revision_tags(cache).items())):
            raise RuntimeError("Completed revised daily artifact or policy differs")
        return metadata
    if destination.exists() or metadata_path.exists():
        raise RuntimeError("Incomplete revised daily artifact; no silent replacement")
    if not source.exists() or not source.with_suffix(".json").exists():
        return None
    fitted = _json(source.with_suffix(".json"))
    learner = fitted.get("learnerMetadata") or {}
    if (fitted.get("trainingCutoffEpoch") != cutoff or fitted.get("maximumTrainingCompleteEpoch", cutoff) >= cutoff
            or fitted.get("trainingRows", 0) < 120 or fitted.get("distinctTrainingTargetDates", 0) < 10
            or fitted.get("fittedAtEpoch", issue_epoch + 1) > issue_epoch
            or learner.get("variant") != "patchtst" or learner.get("epochs") != 30
            or learner.get("normalizerTrainingOnly") is not True
            or fitted.get("dailyPolicySha256") != _sha(cache.HERE / "DAILY_POLICY.json")
            or fitted.get("checkpointSha256") != _sha(source)):
        raise RuntimeError("Original completed daily artifact metadata or hash invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise RuntimeError("Uncompleted artifact adoption exists")
    shutil.copyfile(source, temporary)
    if _sha(temporary) != fitted["checkpointSha256"]:
        raise RuntimeError("Adopted checkpoint byte hash differs")
    temporary.replace(destination)
    info = {**fitted, "checkpoint": str(destination.relative_to(cache.HERE)),
            "adoptedOriginalRuntimeCheckpoint": True,
            "adoptedOriginalMetadataSha256": _sha(source.with_suffix(".json")),
            "adoptedAtEpoch": time.time(), **_revision_tags(cache)}
    cache.original.write(metadata_path, info)
    return info


def adopt_completed_artifact(cache, cutoff, issue_epoch):
    """Serialize byte-identical adoption; no model fitting or old-file writes."""
    with _LOCK:
        return _adopt_completed_artifact(cache, cutoff, issue_epoch)


def background_step(db_path, now=None):
    """New background lifecycle using unchanged sealed daily training math."""
    cache = get_cache()
    now = time.time() if now is None else float(now)
    cutoff = cache._midnight(now)
    adopt_completed_artifact(cache, cutoff, now)
    # This is the sole fitting seam. The original recipe, completed-label and
    # protected-window guards execute unchanged in the isolated module.
    status = cache.step(db_path, int(now))
    if status.get("state") == "skipped" and "trainingRows" not in status:
        # Keep genuine input collection alive when completed training support
        # is insufficient; this separate fallback never fits or invents cases.
        collection = cache.capture_cases(db_path, time.time())
        status = {**status, "collectionAttemptedWithoutModel": True,
                  "modelUnavailableReason": status.get("reason"),
                  "capturedInputGroups": collection["captured"]}
        if collection["captured"]:
            status["state"] = "captured_without_model"
        cache._atomic(cache.CACHE / "STATUS.json", status)
    import patchtst_session_forecast as live
    readiness = live.prepare_runtime(cache, int(time.time()))
    return {**status, **_revision_tags(cache), "productionRuntime": readiness}
