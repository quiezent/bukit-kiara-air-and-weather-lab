"""Independent service worker for the fixed prospective PatchTST experiment.

This module is started once by the existing dashboard service. Training and
compact input collection happen here, never in an HTTP request handler.
"""
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
INTERVAL_SECONDS = 30


def collect_when_fit_unavailable(cache, db_path, status):
    """A missing research model must not stop genuine input collection."""
    if status.get("state") != "skipped" or "trainingRows" in status:
        return status
    # The existing collector still checks the frozen sources, actual receipt
    # clocks and tick grace. This path neither fits nor predicts a model.
    collection = cache.capture_cases(db_path, time.time())
    status = dict(status, collectionAttemptedWithoutModel=True,
                  modelUnavailableReason=status.get("reason"),
                  capturedInputGroups=collection["captured"])
    if collection["captured"]:
        status["state"] = "captured_without_model"
    cache._atomic(cache.CACHE / "STATUS.json", status)
    return status


def run(db_path, stop_event=None):
    # Import the research runtime only in this background thread. The public
    # dashboard can bind and serve requests while Torch initializes or fits.
    forward_dir = HERE / "forward"
    if str(forward_dir) not in sys.path:
        sys.path.insert(0, str(forward_dir))
    import daily_cache

    previous = None
    previous_blend = None
    print("[PatchTST research] Independent forward collection started", flush=True)
    while stop_event is None or not stop_event.is_set():
        try:
            status = daily_cache.step(db_path, int(time.time()))
            status = collect_when_fit_unavailable(daily_cache, db_path, status)
            # The published Afternoon seam only consumes a completed original
            # daily artifact. Runtime verification/loading stays in this worker.
            import patchtst_session_forecast
            patchtst_session_forecast.prepare_runtime(daily_cache, int(time.time()))
            summary = (status.get("state"), status.get("trainingCutoffEpoch"),
                       status.get("reason"))
            if summary != previous:
                print(f"[PatchTST research] {summary}", flush=True)
                previous = summary
        except Exception as error:
            summary = ("error", type(error).__name__, str(error))
            if summary != previous:
                print(f"[PatchTST research] {type(error).__name__}: {error}", flush=True)
                previous = summary
        # This separate failure boundary preserves the original Patch worker
        # and permits noon collection when only its research fit is unavailable.
        try:
            import afternoon_wind61
            blend_status = afternoon_wind61.step(db_path, int(time.time()))
            blend_summary = (blend_status.get('state'), blend_status.get('trainingCutoffEpoch'),
                             blend_status.get('reason'))
            if blend_summary != previous_blend:
                print(f'[Afternoon wind SVR research] {blend_summary}', flush=True)
                previous_blend = blend_summary
        except Exception as error:
            blend_summary = ('error', type(error).__name__, str(error))
            if blend_summary != previous_blend:
                print(f'[Afternoon wind SVR research] {type(error).__name__}: {error}', flush=True)
                previous_blend = blend_summary
        if stop_event is None:
            time.sleep(INTERVAL_SECONDS)
        else:
            stop_event.wait(INTERVAL_SECONDS)
