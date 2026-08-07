"""
Reproducibility manifest for a sweep run: git state, package versions,
hardware, environment, and what got skipped/failed.
results/sweeps/<sweep_id>/manifest.json answers "what exactly produced
these numbers, and can this be reproduced" without digging through chat
history or memory.
"""

import copy
import importlib.metadata
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict


_TRACKED_PACKAGES = [
    "pennylane",
    "pennylane-lightning",
    "dimod",
    "dwave-neal",
    "dwave-system",
    "pyomo",
    "highspy",
    "numpy",
    "scipy",
    "pandas",
    "plotly",
    "networkx",
]


def _run(*args) -> Any:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _git_info() -> Dict[str, Any]:
    commit = _run("rev-parse", "HEAD")
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    dirty = _run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(dirty) if dirty is not None else None,
    }


def _package_versions() -> Dict[str, Any]:
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _gpu_info() -> Any:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out.splitlines() if out else None
    except Exception:
        return None


def _hardware_info() -> Dict[str, Any]:
    return {
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu": _gpu_info(),
    }


def build_manifest(sweep_config: Dict[str, Any], sweep_id: str) -> Dict[str, Any]:
    """Called once at sweep start and written to disk immediately (partial),
    so a crashed sweep still leaves a record of what it intended to run."""
    return {
        "sweep_id": sweep_id,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "last_checkpoint": None,
        # Also buried inside sweep_config.sweep.seed below, but surfaced at
        # the top level so it's visible without digging — see
        # SweepRunner: auto-fills a dwave entry's own params.seed when it
        # doesn't set one, and seeds numpy's global RNG at sweep start. Not
        # a determinism guarantee for every backend (PennyLane has no real
        # seed hook today).
        "global_seed": (sweep_config.get("sweep", {}) or {}).get("seed"),
        "sweep_config": copy.deepcopy(sweep_config),
        "git": _git_info(),
        "packages": _package_versions(),
        "hardware": _hardware_info(),
        "env": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        # Hardware-gated entries the config wanted but that were skipped
        # (reason: "hardware_gated_not_enabled") land here — so a reviewer
        # reading the manifest sees the sweep *intended* the real-hardware
        # run and it was deliberately skipped, not silently missing.
        "skipped": [],
        "failures": [],
        # Populated only when a --resume run picks this manifest back up —
        # each entry is a fresh environment snapshot at the moment of resume,
        # since status could differ from original run
        "resume_events": [],
    }


def finalize_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Call exactly once, when the sweep's run loop reaches its natural end
    (not on every checkpoint) — sets end_time, the actual "this sweep
    finished" signal. Also updates last_checkpoint, since finishing is
    itself the final checkpoint."""
    now = datetime.now(timezone.utc).isoformat()
    manifest["end_time"] = now
    manifest["last_checkpoint"] = now
    return manifest


def checkpoint_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Call after every combo during a run (see SweepRunner._persist()) —
    updates last_checkpoint only. Deliberately does not touch end_time, so
    a manifest read mid-sweep (or after an interruption) correctly shows
    end_time still null."""
    manifest["last_checkpoint"] = datetime.now(timezone.utc).isoformat()
    return manifest


def record_resume_event(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Called when SweepRunner resumes a prior (possibly interrupted) sweep
    — appends a fresh environment snapshot rather than overwriting the
    original manifest's start_time/git/packages/hardware, so the record of
    what *originally* produced most of the results is preserved. Also
    resets end_time back to null: a resumed run is by definition back "in
    progress" until it either reaches finalize_manifest() again or gets
    interrupted again — a stale end_time from a previous (possibly
    incomplete, since resume exists to fill in leftover combos) run would
    be actively misleading here."""
    manifest.setdefault("resume_events", []).append(
        {
            "resumed_at": datetime.now(timezone.utc).isoformat(),
            "git": _git_info(),
            "packages": _package_versions(),
            "hardware": _hardware_info(),
        }
    )
    manifest["end_time"] = None
    return manifest
