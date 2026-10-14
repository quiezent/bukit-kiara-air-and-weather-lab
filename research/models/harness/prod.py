"""Read-only access to the production code via a frozen byte-exact mirror.

The live ``server`` tree is never imported directly: a module that writes
relative to its own ROOT (private/, the DB) would write into production. The
mirror under data/models/prod_src_20261013 is a byte copy taken once; every
later run verifies that the mirror still matches its manifest and reports any
drift of the live server sources (the incumbents are "production as of the
snapshot"). Bytecode writing is disabled before any production import.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

from . import config

MANIFEST = config.PROD_MIRROR / "MIRROR_MANIFEST.json"
_EXCLUDE_DIRS = {"__pycache__", "private", "private_raw", "external_airgradient", "service"}
_EXCLUDE_SUFFIXES = {".db", ".sqlite3", ".db-wal", ".db-shm", ".jsonl", ".exe", ".log"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files():
    for path in sorted(config.SERVER.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(config.SERVER)
        if any(part in _EXCLUDE_DIRS for part in relative.parts[:-1]):
            continue
        if path.suffix.lower() in _EXCLUDE_SUFFIXES or path.name.endswith((".pending", ".tmp")):
            continue
        yield relative


def ensure_mirror(refresh: bool = False) -> dict:
    """Create the mirror once (byte copies); later calls only verify it."""
    if MANIFEST.exists() and not refresh:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        return manifest
    if config.PROD_MIRROR.exists():
        shutil.rmtree(config.PROD_MIRROR)
    files = {}
    for relative in _source_files():
        target = config.PROD_MIRROR / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.SERVER / relative, target)
        files[relative.as_posix()] = _sha(target)
    manifest = {"source": str(config.SERVER), "fileCount": len(files), "files": files,
                "note": "Byte copy of production code/research assets; DB, private/, logs excluded."}
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def drift(manifest: dict | None = None) -> dict:
    """Compare the live server's top-level .py files with the frozen mirror."""
    manifest = manifest or ensure_mirror()
    changed = []
    for relative, digest in manifest["files"].items():
        if "/" in relative or not relative.endswith(".py"):
            continue
        live = config.SERVER / relative
        if not live.exists() or _sha(live) != digest:
            changed.append(relative)
    return {"changedTopLevelModules": changed}


_IMPORTED = False


def setup() -> Path:
    """Put the mirror (not the live server) on sys.path; never write bytecode."""
    global _IMPORTED
    sys.dont_write_bytecode = True
    ensure_mirror()
    root = str(config.PROD_MIRROR)
    if str(config.SERVER) in sys.path:
        raise RuntimeError("Live server directory must not be on sys.path")
    if root not in sys.path:
        sys.path.insert(0, root)
    _IMPORTED = True
    return config.PROD_MIRROR


def module(name: str):
    setup()
    return importlib.import_module(name)
