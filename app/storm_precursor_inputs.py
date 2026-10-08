"""Collect wider radar research inputs through the existing bounded worker."""
from pathlib import Path

VERSION = "ttdi_storm_precursor_archive_v1"


def poll_once(legacy_path):
    """Preserve radar status and cadence; retain receipt-stamped event features."""
    from research.event_precursor_sources.radar_precursors import poll_once as collect

    path = Path(legacy_path)
    return collect(path, feature_output_path=path.with_name("radar-event-precursors.jsonl"))
