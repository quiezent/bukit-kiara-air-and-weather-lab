"""Receipt-stamped nearby PM collection for private model research.

The independent service thread archives provider-corrected measurements. It
does not read targets, train a model, or alter any published forecast numbers.
"""
from pathlib import Path
import time

VERSION='airgradient_neighbor_receipt_archive_v1'
ARCHIVE_PATH=Path(__file__).resolve().parent/'research'/'external_airgradient'/'neighbors.sqlite3'


def collector():
    """Keep input collection alive independently of browsers and Codex."""
    while True:
        try:
            from research.external_airgradient.neighbor_history import collect_current_forever
            collect_current_forever(ARCHIVE_PATH,interval=300)
        except Exception as error:
            print(f'[Neighbor PM collector] {type(error).__name__}: {error}',flush=True)
            time.sleep(300)
