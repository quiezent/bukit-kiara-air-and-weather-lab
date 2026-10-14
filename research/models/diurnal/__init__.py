"""Diurnal / time-of-day candidate (conditional climatology and time-feature HGB)."""
import os

# One OpenMP thread: on this shared, throttled laptop HistGradientBoosting with 2
# threads spin-waits and ran 10-20x slower than with 1 (measured 2026-10-14).
# Set before harness.config (which only sets a default of 2) and sklearn load.
os.environ.setdefault("OMP_NUM_THREADS", "1")
