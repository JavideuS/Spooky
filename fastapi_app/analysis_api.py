"""
Read-only analysis API: serves aggregated benchmark-sweep results — tables,
Plotly figures, and a per-instance-class solver recommendation — for the
demo's "Benchmarks" tab.

Data source is a directory of sweep dirs: SPOOKY_BENCHMARKS_DIR, default
<repo>/results/sweeps. Each sweep dir is one produced by
quantum.benchmark.sweep_runner (index.json + manifest.json + per-combo
benchmark JSONs). Nested layouts (results/sweeps/CML/<id>/) are discovered
recursively.

Nothing here runs a sweep or a solve. aggregate_sweep() is recomputed from
the raw benchmark JSONs on the first request per sweep and cached in-process
for the lifetime of the process; the CSVs it writes as a side effect go to a
scratch cache dir, never back into the (possibly read-only, e.g. a Hugging
Face dataset snapshot) source tree.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

import quantum
import registry
from quantum.benchmark.analysis import plots as sweep_plots
from quantum.benchmark.analysis.aggregate import (
    _valid_mask,
    aggregate_sweep,
    run_statistical_tests,
)

logger = logging.getLogger("spooky.api.analysis")

QUANTUM_ROOT = Path(quantum.__file__).resolve().parent

# Where published sweep dirs live. Local dev: the repo's results/sweeps. In a
# deployed Space this points at the huggingface_hub snapshot_download cache.
BENCHMARKS_DIR = Path(
    os.environ.get(
        "SPOOKY_BENCHMARKS_DIR", str(QUANTUM_ROOT.parent / "results" / "sweeps")
    )
).expanduser()

# aggregate_sweep() writes CSVs; send them here so a read-only source tree
# (an HF snapshot) still works. Mirrors registry.MAP_CACHE_ROOT.
_CACHE_ROOT = (
    Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    / "spooky"
    / "benchmarks"
)

router = APIRouter(prefix="/v1/analysis", tags=["analysis"])

_GRID_RE = re.compile(r"(\d+x\d+)")

# sweep_id -> aggregate_sweep() result (dict of DataFrames), filled lazily.
_tables_cache: Dict[str, Dict[str, pd.DataFrame]] = {}


# --------------------------------------------------------------------------
# Sweep discovery + catalog
# --------------------------------------------------------------------------


def _iter_sweep_dirs() -> List[Path]:
    if not BENCHMARKS_DIR.exists():
        return []
    found = []
    for index_path in BENCHMARKS_DIR.rglob("index.json"):
        sweep_dir = index_path.parent
        if (sweep_dir / "manifest.json").exists():
            found.append(sweep_dir)
    return found


@lru_cache(maxsize=1)
def _sweep_dir_map() -> Dict[str, Path]:
    """sweep_id -> dir. sweep_id is the dir basename; on the near-impossible
    collision between two trees the first discovered wins and the clash is
    logged. Cleared by GET /v1/analysis/sweeps?refresh=true."""
    mapping: Dict[str, Path] = {}
    for sweep_dir in _iter_sweep_dirs():
        if sweep_dir.name in mapping:
            logger.warning(
                "Duplicate sweep_id %s (%s vs %s) — keeping the first",
                sweep_dir.name,
                mapping[sweep_dir.name],
                sweep_dir,
            )
            continue
        mapping[sweep_dir.name] = sweep_dir
    return mapping


def _resolve_sweep_dir(sweep_id: str) -> Path:
    sweep_dir = _sweep_dir_map().get(sweep_id)
    if sweep_dir is None:
        raise HTTPException(404, f"Unknown sweep_id: {sweep_id}")
    return sweep_dir


def _catalog_entry(sweep_dir: Path) -> Dict[str, Any]:
    with open(sweep_dir / "index.json", encoding="utf-8") as f:
        index = json.load(f)
    with open(sweep_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    completed = [e for e in index if not e.get("dry_run") and e.get("benchmark_json")]
    instances = sorted({e["instance"] for e in index})
    grid_sizes = sorted({m.group(1) for i in instances if (m := _GRID_RE.search(i))})
    git = manifest.get("git") or {}
    return {
        "sweep_id": sweep_dir.name,
        "start_time": manifest.get("start_time"),
        "end_time": manifest.get("end_time"),
        "finished": manifest.get("end_time") is not None,
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "git_dirty": git.get("dirty"),
        "n_planned": len(index),
        "n_completed": len(completed),
        "solvers": sorted({e["solver"] for e in index}),
        "backends": sorted({e["backend"] for e in index}),
        "problems": sorted({e["problem"] for e in index}),
        "instances": instances,
        "grid_sizes": grid_sizes,
    }


# --------------------------------------------------------------------------
# Aggregation (cached) + serialization helpers
# --------------------------------------------------------------------------


def _tables(sweep_id: str) -> Dict[str, pd.DataFrame]:
    if sweep_id not in _tables_cache:
        sweep_dir = _resolve_sweep_dir(sweep_id)
        try:
            _tables_cache[sweep_id] = aggregate_sweep(
                str(sweep_dir), output_dir=str(_CACHE_ROOT / sweep_id)
            )
        except FileNotFoundError as exc:
            raise HTTPException(422, f"Sweep {sweep_id} is incomplete: {exc}")
        except ValueError as exc:
            raise HTTPException(422, f"Sweep {sweep_id} has no usable runs: {exc}")
    return _tables_cache[sweep_id]


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """DataFrame -> list of plain dicts, NaN -> null. Via df.to_json because
    FastAPI's default encoder emits invalid bare `NaN` tokens for a raw
    float nan, and to_json also normalizes numpy scalar dtypes."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _fnone(value: Any) -> Optional[float]:
    return None if value is None or pd.isna(value) else float(value)


def _filter_runs(
    df: pd.DataFrame,
    *,
    solver: Optional[str] = None,
    grid_size: Optional[str] = None,
    problem: Optional[str] = None,
    num_robots: Optional[int] = None,
    preprocess: Optional[str] = None,
) -> pd.DataFrame:
    if solver:
        df = df[df["solver_name"] == solver]
    if grid_size:
        df = df[df["grid_size"] == grid_size]
    if problem:
        df = df[df["problem_name"] == problem]
    if num_robots is not None:
        df = df[df["num_robots"] == num_robots]
    if preprocess:
        df = df[df["preprocess"].astype(str) == preprocess]
    return df


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@router.get("/sweeps")
def list_sweeps(refresh: bool = False):
    """Catalog of every discoverable sweep, newest first. refresh=true
    re-scans SPOOKY_BENCHMARKS_DIR and drops the aggregation cache — use it
    after dropping new sweep data in without restarting the process."""
    if refresh:
        _sweep_dir_map.cache_clear()
        _tables_cache.clear()
    entries = [_catalog_entry(d) for d in _sweep_dir_map().values()]
    entries.sort(key=lambda e: e.get("start_time") or "", reverse=True)
    return {
        "benchmarks_dir": str(BENCHMARKS_DIR),
        "sweep_count": len(entries),
        "sweeps": entries,
    }


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str):
    """One catalog entry plus the full reproducibility manifest."""
    sweep_dir = _resolve_sweep_dir(sweep_id)
    entry = _catalog_entry(sweep_dir)
    with open(sweep_dir / "manifest.json", encoding="utf-8") as f:
        entry["manifest"] = json.load(f)
    return entry


@router.get("/sweeps/{sweep_id}/instances")
def list_instances(sweep_id: str):
    """The (instance, problem, grid_size, num_robots) combos actually present
    in the sweep — drives the Benchmarks tab's instance picker and its map
    thumbnails. map_id is the registry key (for /v1/maps/{map_id}/preview)
    when the instance's map is one of the curated maps, else null."""
    df = _tables(sweep_id)["runs_long"]
    known_maps = set(registry.list_maps().keys())
    grouped = (
        df.groupby(
            ["instance_map", "problem_name", "grid_size", "num_robots"], dropna=False
        )
        .agg(
            solvers=("solver_name", lambda s: sorted(set(s))),
            n_runs=("run_id", "size"),
        )
        .reset_index()
    )
    instances = []
    for row in grouped.itertuples(index=False):
        map_basename = str(row.instance_map).split("/")[-1]
        instances.append(
            {
                "instance_map": row.instance_map,
                "map_id": map_basename if map_basename in known_maps else None,
                "problem_name": row.problem_name,
                "grid_size": row.grid_size,
                "num_robots": int(row.num_robots) if pd.notna(row.num_robots) else None,
                "solvers": list(row.solvers),
                "n_runs": int(row.n_runs),
            }
        )
    return {
        "sweep_id": sweep_id,
        "instance_count": len(instances),
        "instances": instances,
    }


@router.get("/sweeps/{sweep_id}/summary")
def sweep_summary(sweep_id: str):
    """The aggregated tables (see quantum.benchmark.analysis.aggregate):
    per-solver success/reduction, the paired Wilcoxon tests, failure causes,
    and the per-configuration energy diagnostics."""
    tables = _tables(sweep_id)
    return {
        "sweep_id": sweep_id,
        "summary_by_solver": _records(tables["summary_by_solver"]),
        "statistical_tests": _records(tables["statistical_tests"]),
        "failure_causes": _records(tables["failure_causes"]),
        "energy_diagnostics": _records(tables["energy_diagnostics"]),
    }


@router.get("/sweeps/{sweep_id}/runs")
def sweep_runs(
    sweep_id: str,
    solver: Optional[str] = None,
    grid_size: Optional[str] = None,
    problem: Optional[str] = None,
    num_robots: Optional[int] = None,
    preprocess: Optional[str] = None,
    valid: Optional[bool] = None,
):
    """The long-format per-run table, optionally filtered."""
    df = _filter_runs(
        _tables(sweep_id)["runs_long"],
        solver=solver,
        grid_size=grid_size,
        problem=problem,
        num_robots=num_robots,
        preprocess=preprocess,
    )
    if valid is not None:
        df = df[_valid_mask(df) == valid]
    return {"sweep_id": sweep_id, "run_count": len(df), "runs": _records(df)}


def _filter_robot_stats(robot_df: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """robot_statistics_long has no grid_size / num_robots column, so it can't
    be sliced by _filter_runs directly. Keep only the per-robot rows whose
    (instance_map, problem_name, solver_name) survived the runs_long filter."""
    if robot_df.empty:
        return robot_df
    keys = runs[["instance_map", "problem_name", "solver_name"]].drop_duplicates()
    return robot_df.merge(
        keys, on=["instance_map", "problem_name", "solver_name"], how="inner"
    )


_PLOT_BUILDERS = {
    "scaling": lambda t: sweep_plots.plot_scaling(t["runs_long"]),
    "success_rate": lambda t: sweep_plots.plot_success_rate_bars(t["runs_long"]),
    "variable_reduction": lambda t: sweep_plots.plot_variable_reduction(t["runs_long"]),
    "energy_excess": lambda t: sweep_plots.plot_energy_excess(
        t["runs_long"], boxpoints="outliers"
    ),
    "path_efficiency": lambda t: sweep_plots.plot_path_efficiency(
        t["robot_statistics_long"], boxpoints="outliers"
    ),
}

# scaling is a cross-size view by construction (x-axis is problem scale) —
# filtering it to one grid_size / num_robots collapses it, so it always runs
# on the whole sweep. The rest honour the instance-class filter.
_UNFILTERABLE_PLOTS = {"scaling"}


@router.get("/sweeps/{sweep_id}/plots/{name}")
def sweep_plot(
    sweep_id: str,
    name: str,
    grid_size: Optional[str] = None,
    problem: Optional[str] = None,
    num_robots: Optional[int] = None,
):
    """A single Plotly figure as {data, layout} JSON — the demo loads
    plotly.js itself and calls Plotly.newPlot with this. All plots except
    'scaling' accept the same grid_size / problem / num_robots filter as
    /recommend."""
    if name not in _PLOT_BUILDERS:
        raise HTTPException(
            404, f"Unknown plot: {name}. Available: {sorted(_PLOT_BUILDERS)}"
        )
    tables = _tables(sweep_id)
    if name == "path_efficiency" and tables["robot_statistics_long"].empty:
        raise HTTPException(
            404,
            "No per-robot path-efficiency data in this sweep "
            "(needs a BenchmarkRunner level>=2 run).",
        )

    if name not in _UNFILTERABLE_PLOTS and (
        grid_size or problem or num_robots is not None
    ):
        runs = _filter_runs(
            tables["runs_long"],
            grid_size=grid_size,
            problem=problem,
            num_robots=num_robots,
        )
        if runs.empty:
            raise HTTPException(404, "No runs match that filter for this plot.")
        tables = {
            **tables,
            "runs_long": runs,
            "robot_statistics_long": _filter_robot_stats(
                tables["robot_statistics_long"], runs
            ),
        }

    try:
        fig = _PLOT_BUILDERS[name](tables)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, f"Cannot build plot '{name}' for {sweep_id}: {exc}")
    return json.loads(fig.to_json())


@router.get("/sweeps/{sweep_id}/recommend")
def recommend(
    sweep_id: str,
    grid_size: Optional[str] = None,
    problem: Optional[str] = None,
    num_robots: Optional[int] = None,
):
    """Per-solver aggregates over one instance class, plus the paired
    statistical tests recomputed on that slice and a plain-language verdict.

    Deliberately not a single "winner": energy is not comparable across
    solvers, and success rates carry survivorship bias — the response
    surfaces n_instances / n_runs alongside so the caller can see how thin
    the comparison is."""
    sub = _filter_runs(
        _tables(sweep_id)["runs_long"],
        grid_size=grid_size,
        problem=problem,
        num_robots=num_robots,
    )
    if sub.empty:
        raise HTTPException(404, "No runs match that instance class.")

    sub = sub.assign(_valid=_valid_mask(sub))
    valid = sub[sub["_valid"]]

    solvers = []
    for solver, group in sub.groupby("solver_name"):
        valid_group = valid[valid["solver_name"] == solver]
        solvers.append(
            {
                "solver": solver,
                "backend": group["backend"].iloc[0],
                "n_runs": int(len(group)),
                "n_instances": int(
                    group[["instance_map", "problem_name"]].drop_duplicates().shape[0]
                ),
                "success_rate": float(group["_valid"].mean()),
                "mean_time_sec": _fnone(valid_group["execution_time_sec"].mean()),
                "median_time_sec": _fnone(valid_group["execution_time_sec"].median()),
                "mean_path_efficiency": _fnone(
                    valid_group["avg_path_efficiency"].mean()
                ),
                "median_path_efficiency": _fnone(
                    valid_group["avg_path_efficiency"].median()
                ),
            }
        )
    solvers.sort(
        key=lambda r: (
            -r["success_rate"],
            r["mean_time_sec"] if r["mean_time_sec"] is not None else float("inf"),
        )
    )

    try:
        tests = (
            run_statistical_tests(sub)
            if sub["solver_name"].nunique() > 1
            else pd.DataFrame()
        )
    except Exception as exc:  # defensive: a thin slice shouldn't 500 the route
        logger.warning("run_statistical_tests failed for %s: %s", sweep_id, exc)
        tests = pd.DataFrame()

    slice_desc = (
        ", ".join(
            p
            for p in (
                grid_size,
                f"{num_robots} robots" if num_robots is not None else None,
                problem,
            )
            if p
        )
        or "the whole sweep"
    )

    verdict = None
    if solvers:
        best = solvers[0]
        verdict = (
            f"On {slice_desc}, {best['solver']} had the highest success rate "
            f"({best['success_rate']:.0%})"
        )
        if best["mean_time_sec"] is not None:
            verdict += f" at {best['mean_time_sec']:.3g}s mean solve time"
        verdict += (
            ". Energy is not comparable across solvers — path efficiency and "
            "the paired tests are the cross-solver quality measures, and "
            "success rates carry survivorship bias (check n_instances / n_runs)."
        )

    return {
        "sweep_id": sweep_id,
        "instance_class": {
            "grid_size": grid_size,
            "num_robots": num_robots,
            "problem": problem,
        },
        "solvers": solvers,
        "statistical_tests": _records(tests) if not tests.empty else [],
        "verdict": verdict,
    }


@router.get("/sweeps/{sweep_id}/by-instance")
def by_instance(
    sweep_id: str,
    grid_size: Optional[str] = None,
    problem: Optional[str] = None,
    num_robots: Optional[int] = None,
    solver: Optional[str] = None,
):
    """One row per (map, problem, solver): success rate, valid/total run
    counts, mean solve time, and the invalid-cause breakdown — 'pre_processing'
    (BFS / diagonal fixing pinned a conflict before the solver ran) vs
    'solver_sampling' (the solver returned a violating bitstring). This is the
    per-map "which fails, and why" table; no plot."""
    df = _filter_runs(
        _tables(sweep_id)["runs_long"],
        solver=solver,
        grid_size=grid_size,
        problem=problem,
        num_robots=num_robots,
    )
    if df.empty:
        raise HTTPException(404, "No runs match that instance class.")

    df = df.assign(_valid=_valid_mask(df))
    rows = []
    for (instance_map, prob, solver_name), group in df.groupby(
        ["instance_map", "problem_name", "solver_name"], sort=False
    ):
        invalid = group[~group["_valid"]]
        causes = invalid["invalid_cause"].fillna("unrecorded").value_counts().to_dict()
        valid_group = group[group["_valid"]]
        robots_val = group["num_robots"].iloc[0]
        rows.append(
            {
                "instance_map": instance_map,
                "map": str(instance_map).split("/")[-1],
                "problem_name": prob,
                "solver": solver_name,
                "backend": group["backend"].iloc[0],
                "grid_size": group["grid_size"].iloc[0],
                "num_robots": int(robots_val) if pd.notna(robots_val) else None,
                "n_runs": int(len(group)),
                "n_valid": int(group["_valid"].sum()),
                "success_rate": float(group["_valid"].mean()),
                "mean_time_sec": _fnone(valid_group["execution_time_sec"].mean()),
                "failure_causes": {str(k): int(v) for k, v in causes.items()},
            }
        )
    rows.sort(key=lambda r: (r["map"], r["problem_name"], -r["success_rate"]))
    return {"sweep_id": sweep_id, "row_count": len(rows), "rows": rows}
