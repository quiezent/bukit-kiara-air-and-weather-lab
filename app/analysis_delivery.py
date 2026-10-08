"""Nonblocking delivery of immutable, timestamped forecast snapshots.

Only the background producer calculates forecasts. HTTP readers never acquire a
model lock, start a fit, or relabel a saved issue/target. No telemetry is stored.
"""
from collections import OrderedDict
import copy
import threading
import time


_UNSET = object()


class CoalescingDiagnosticQueue:
    """Keep at most one waiting diagnostic snapshot behind the active fit."""

    def __init__(self):
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.pending = None

    def submit(self, job):
        with self.lock:
            displaced = self.pending
            self.pending = job
            self.wake.set()
            return displaced

    def take(self, timeout=None):
        self.wake.wait(timeout)
        with self.lock:
            job = self.pending
            self.pending = None
            self.wake.clear()
            return job


class AnalysisDelivery:
    def __init__(self, refresh_seconds=60, maximum_age_seconds=600,
                 retry_seconds=30, max_keys=4, clock=time.time):
        self.refresh_seconds = refresh_seconds
        self.maximum_age_seconds = maximum_age_seconds
        self.retry_seconds = retry_seconds
        self.max_keys = max_keys
        self.clock = clock
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.entries = OrderedDict()
        self.pending = OrderedDict()
        self.active_key = None
        self.phase = None

    def _entry(self, key):
        if key not in self.entries:
            while len(self.entries) >= self.max_keys:
                victim = next((k for k in self.entries if k != self.active_key), None)
                if victim is None:
                    break
                self.entries.pop(victim)
                self.pending.pop(victim, None)
            self.entries[key] = {"result": None, "attempt": None,
                                 "error": None, "diagnostics": "pending"}
        self.entries.move_to_end(key)
        return self.entries[key]

    def _schedule(self, key, now):
        entry = self._entry(key)
        if key == self.active_key or key in self.pending:
            return entry
        result = entry["result"]
        issued = (result or {}).get("forecastIssuedEpoch")
        due = result is None or issued is None or now - issued >= self.refresh_seconds
        if result is not None and issued is None and entry["attempt"] is not None:
            due = now - entry["attempt"] >= self.refresh_seconds
        if entry["error"] and entry["attempt"] is not None:
            due = due and now - entry["attempt"] >= self.retry_seconds
        if due:
            self.pending[key] = None
            self.wake.set()
        return entry

    def schedule(self, key):
        with self.lock:
            self._schedule(key, self.clock())

    def take_job(self):
        """Claim a single job; repeated readers only coalesce requests."""
        with self.lock:
            if self.active_key is not None or not self.pending:
                return None
            key, _ = self.pending.popitem(last=False)
            self.active_key, self.phase = key, "primary"
            self.entries[key]["attempt"] = self.clock()
            self.entries[key]["error"] = None
            return key

    def publish_primary(self, key, result, *, diagnostics="pending"):
        # Copy before taking the coordination lock: publication is atomic and
        # readers never share mutable dictionaries with enrichment code.
        saved = copy.deepcopy(result)
        with self.lock:
            if self.active_key != key:
                raise RuntimeError("Only the active producer can publish")
            entry = self.entries[key]
            entry.update(result=saved, error=None,
                         diagnostics=diagnostics if saved.get("available") else "complete")
            self.phase = "diagnostics" if diagnostics == "pending" else None

    def publish_diagnostics(self, key, result, *, stale_ok=False):
        saved = copy.deepcopy(result)
        with self.lock:
            if not stale_ok and self.active_key != key:
                raise RuntimeError("Only the active producer can enrich")
            entry = self.entries.get(key)
            if entry is None:
                if stale_ok:
                    return False
                raise RuntimeError("Diagnostic entry was evicted")
            if (entry["result"] or {}).get("forecastIssuedEpoch") != saved.get("forecastIssuedEpoch"):
                if stale_ok:
                    return False
                raise ValueError("Diagnostics cannot relabel a forecast issue")
            entry["result"] = saved
            entry["diagnostics"] = "complete"
            return True

    def fail(self, key, error, diagnostics=False, *, expected_issue_epoch=_UNSET):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                return
            if diagnostics:
                if (expected_issue_epoch is not _UNSET
                        and (entry["result"] or {}).get("forecastIssuedEpoch") != expected_issue_epoch):
                    return
                entry["diagnostics"] = "error"
            else:
                entry["error"] = str(error)
                # Backoff starts on failure, not before a potentially slow fit.
                entry["attempt"] = self.clock()

    def finish(self, key):
        with self.lock:
            if self.active_key == key:
                self.active_key, self.phase = None, None

    def response(self, key):
        now = self.clock()
        with self.lock:
            entry = self._schedule(key, now)
            saved = entry["result"]
            issued = (saved or {}).get("forecastIssuedEpoch")
            age = max(0, now - issued) if issued is not None else None
            expired = age is not None and age > self.maximum_age_seconds
            updating = key in self.pending or (self.active_key == key and self.phase == "primary")
            state = ("expired" if expired else "error" if entry["error"]
                     else "initializing" if saved is None
                     else "updating" if updating else "ready")
            metadata = {
                "state": state, "updating": updating,
                "forecastAgeSeconds": round(age, 1) if age is not None else None,
                "servedAtEpoch": int(now), "maximumAgeSeconds": self.maximum_age_seconds,
                "error": entry["error"],
                "retryAfterSeconds": 5 if updating else self.retry_seconds,
                "diagnostics": entry["diagnostics"],
            }
        # Saved results are replaced, never mutated. Deep-copy outside the lock
        # keeps simultaneous HTTP clients from queueing behind one another.
        if saved is None or expired:
            response = {
                "available": False,
                "message": ("Forecast expired; preparing an update." if expired
                            else "Forecast update failed; retrying." if metadata["error"]
                            else "Preparing forecast…"),
                "delivery": metadata,
            }
            if saved is not None:
                # Provenance is safe to retain after points/targets expire.
                response["forecastIssuedEpoch"] = saved.get("forecastIssuedEpoch")
                response["computation"] = copy.deepcopy(saved.get("computation") or {})
            return response
        result = copy.deepcopy(saved)
        result["delivery"] = metadata
        return result
