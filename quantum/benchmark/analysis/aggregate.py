"""
pandas/scipy aggregation for a sweep's results.

Reads index.json, flattens every individual run into one row of
a long-format DataFrame, and computes the derived columns/tables
that later the plots and LaTeX export read from.

Every function here is a plain, independently-importable function (not
CLI-only) so a future FastAPI endpoint could call aggregate_sweep()
directly — see run_aggregate.py for the CLI wrapper.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

# Wilcoxon signed-rank has essentially no discriminatory power below this
# many paired instances (commonly cited minimum for the test to be
# meaningfully computable at all)
_MIN_PAIRS_FOR_WILCOXON = 6


def _robot_efficiency_summary(solution_statistics: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """(avg, min) path_efficiency across a run's robots, or (None, None) if
    solution_statistics wasn't recorded (needs BenchmarkRunner level>=2 —
    see quantum.benchmark.benchmark._compute_solution_statistics)."""
    robot_stats = (solution_statistics or {}).get("robot_statistics") or {}
    efficiencies = [
        rs["path_efficiency"] for rs in robot_stats.values() if "path_efficiency" in rs
    ]
    if not efficiencies:
        return None, None
    return sum(efficiencies) / len(efficiencies), min(efficiencies)


def load_sweep(sweep_dir: str) -> pd.DataFrame:
    """Long-format DataFrame, one row per individual solver run."""
    sweep_dir = Path(sweep_dir)
    index_path = sweep_dir / "index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    rows: List[Dict[str, Any]] = []
    for entry in index:
        if entry.get("dry_run") or not entry.get("benchmark_json"):
            continue
        with open(entry["benchmark_json"], "r", encoding="utf-8") as f:
            data = json.load(f)

        problem_meta = data["metadata"]["problem"]
        num_robots = len(problem_meta.get("robots", {}) or {})
        grid = problem_meta.get("grid")
        graph = problem_meta.get("graph")
        if grid:
            grid_size = f"{grid['M']}x{grid['N']}"
        elif graph:
            grid_size = f"{len(graph.get('nodes', []) or [])}nodes"
        else:
            grid_size = None

        for run in data["runs"]:
            var_stats = run.get("variable_stats", {}) or {}
            solution_stats = run.get("solution_statistics") or {}
            avg_efficiency, min_efficiency = _robot_efficiency_summary(solution_stats)
            rows.append(
                {
                    "sweep_id": sweep_dir.name,
                    "instance_map": entry["instance"],
                    "grid_size": grid_size,
                    "problem_name": entry["problem"],
                    "num_robots": num_robots,
                    "solver_name": entry["solver"],
                    "backend": entry["backend"],
                    "device": entry.get("device"),
                    "penalty_set": entry.get("penalty_set"),
                    "preprocess": entry["preprocess"],
                    "run_id": run.get("run_id"),
                    "valid": run.get("valid"),
                    "energy": run.get("energy"),
                    "execution_time_sec": run.get("execution_time_sec"),
                    "num_windows": var_stats.get("num_windows"),
                    "total_initial_variables": var_stats.get("total_initial_variables"),
                    "total_variables_reduced": var_stats.get("total_variables_reduced"),
                    "total_final_variables": var_stats.get("total_final_variables"),
                    "average_reduction_ratio": var_stats.get("average_reduction_ratio"),
                    "termination_condition": run.get("termination_condition"),
                    "avg_path_efficiency": avg_efficiency,
                    "min_path_efficiency": min_efficiency,
                    "robot_success_rate": solution_stats.get("success_rate"),
                    "timestamp": run.get("timestamp"),
                }
            )

    return pd.DataFrame(rows)


def load_robot_statistics(sweep_dir: str) -> pd.DataFrame:
    """Long-format DataFrame, one row per (run, robot) — the per-robot
    detail behind load_sweep()'s avg_path_efficiency/min_path_efficiency
    summary columns (see quantum.benchmark.benchmark._compute_solution_statistics).
    Empty (but correctly shaped) if no run in the sweep was benchmarked at
    level>=2, since solution_statistics only exists there."""
    sweep_dir = Path(sweep_dir)
    index_path = sweep_dir / "index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    columns = [
        "sweep_id", "instance_map", "problem_name", "solver_name", "backend",
        "preprocess", "run_id", "robot_id", "path_length", "moves_taken",
        "optimal_path_length", "path_efficiency", "goal_reached",
        "validation_passed", "priority",
    ]
    rows: List[Dict[str, Any]] = []
    for entry in index:
        if entry.get("dry_run") or not entry.get("benchmark_json"):
            continue
        with open(entry["benchmark_json"], "r", encoding="utf-8") as f:
            data = json.load(f)

        for run in data["runs"]:
            robot_stats = (run.get("solution_statistics") or {}).get(
                "robot_statistics"
            ) or {}
            for robot_id, rs in robot_stats.items():
                rows.append(
                    {
                        "sweep_id": sweep_dir.name,
                        "instance_map": entry["instance"],
                        "problem_name": entry["problem"],
                        "solver_name": entry["solver"],
                        "backend": entry["backend"],
                        "preprocess": entry["preprocess"],
                        "run_id": run.get("run_id"),
                        "robot_id": robot_id,
                        "path_length": rs.get("path_length"),
                        "moves_taken": rs.get("moves_taken"),
                        "optimal_path_length": rs.get("optimal_path_length"),
                        "path_efficiency": rs.get("path_efficiency"),
                        "goal_reached": rs.get("goal_reached"),
                        "validation_passed": rs.get("validation_passed"),
                        "priority": rs.get("priority"),
                    }
                )

    return pd.DataFrame(rows, columns=columns)


def compute_optimality_gap(
    df: pd.DataFrame, reference_solver: str = "ilp"
) -> pd.DataFrame:
    """Adds 'reference_energy', 'optimality_gap', 'reference_missing' columns.

    Reference = mean energy of `reference_solver` rows *with
    termination_condition == "optimal"* only, per (instance_map,
    problem_name) group — never silently substitutes an unproven/timed-out
    result (or a different solver entirely) as ground truth. Rows for an
    instance with no proven-optimal reference get reference_missing=True
    and a NaN gap rather than a misleading comparison.

    gap = (energy - reference) / reference, computed only for valid==True
    rows (an invalid/infeasible run's energy isn't a comparable quantity).
    """
    df = df.copy()
    proven_optimal = df[
        (df["backend"] == reference_solver) & (df["termination_condition"] == "optimal")
    ]
    reference = (
        proven_optimal.groupby(["instance_map", "problem_name"])["energy"]
        .mean()
        .rename("reference_energy")
    )
    df = df.merge(reference, on=["instance_map", "problem_name"], how="left")
    df["reference_missing"] = df["reference_energy"].isna()

    df["optimality_gap"] = np.nan
    comparable = df["valid"] & ~df["reference_missing"] & (df["reference_energy"] != 0)
    df.loc[comparable, "optimality_gap"] = (
        df.loc[comparable, "energy"] - df.loc[comparable, "reference_energy"]
    ) / df.loc[comparable, "reference_energy"]
    return df


def compute_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["instance_map", "problem_name", "solver_name", "preprocess"])[
            "valid"
        ]
        .mean()
        .rename("success_rate")
        .reset_index()
    )


def compute_variable_reduction_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Groups the already-computed average_reduction_ratio column — no new
    computation, BenchmarkRunner already aggregates this per run from each
    solver's own window_stats/bfs_stats."""
    return (
        df.groupby(["instance_map", "problem_name", "solver_name", "preprocess"])
        .agg(
            mean_reduction_ratio=("average_reduction_ratio", "mean"),
            mean_initial_variables=("total_initial_variables", "mean"),
            mean_final_variables=("total_final_variables", "mean"),
        )
        .reset_index()
    )


def run_statistical_tests(
    df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
    metrics: Tuple[str, ...] = ("execution_time_sec", "optimality_gap"),
) -> pd.DataFrame:
    """Wilcoxon signed-rank test per (solver_a, solver_b, metric), with
    Benjamini-Hochberg FDR correction applied across the whole family of
    tests run in this call.

    Pairs *by instance*, not by raw run: for each solver, takes its
    per-instance mean of the metric across (instance_map, problem_name),
    then tests the paired per-instance values against the other solver.
    Raw per-run values from a stochastic solver on the same instance aren't
    independent draws suitable for a cross-solver paired test, only
    per-instance summaries are. Only instances both solvers share (and,
    for optimality_gap, that have a comparable, non-NaN value) are used;
    n_pairs reports exactly how many that was.

    Comparing k solvers x len(metrics) generates many p-values in one call;
    reporting them uncorrected inflates the family-wise false-positive
    rate. p_value is the raw Wilcoxon result; p_value_bh is the
    Benjamini-Hochberg-adjusted one (scipy.stats.false_discovery_control)
    computed over every row in this result that has a real p-value — use
    p_value_bh for any "is this difference significant" claim, not p_value.

    A pair with fewer than _MIN_PAIRS_FOR_WILCOXON shared instances gets
    p_value=None (and note="insufficient_data") rather than a computed-but-
    statistically-meaningless result."""
    if pairs is None:
        solvers = sorted(df["solver_name"].unique())
        baselines = [s for s in ("ilp", "cbs") if s in df["backend"].unique()]
        baseline_names = (
            df[df["backend"].isin(baselines)]["solver_name"].unique().tolist()
        )
        pairs = [(a, b) for a in solvers for b in baseline_names if a != b]
        pairs = sorted(set(tuple(sorted(p)) for p in pairs))

    rows = []
    for solver_a, solver_b in pairs:
        for metric in metrics:
            per_instance = (
                df[df["valid"]]
                .groupby(["instance_map", "problem_name", "solver_name"])[metric]
                .mean()
                .unstack("solver_name")
            )
            if (
                solver_a not in per_instance.columns
                or solver_b not in per_instance.columns
            ):
                continue
            paired = per_instance[[solver_a, solver_b]].dropna()
            n_pairs = len(paired)
            if n_pairs < _MIN_PAIRS_FOR_WILCOXON:
                rows.append(
                    {
                        "solver_a": solver_a,
                        "solver_b": solver_b,
                        "metric": metric,
                        "n_pairs": n_pairs,
                        "statistic": None,
                        "p_value": None,
                        "note": (
                            f"insufficient_data (need >= {_MIN_PAIRS_FOR_WILCOXON} "
                            "paired instances)"
                        ),
                    }
                )
                continue
            try:
                statistic, p_value = stats.wilcoxon(paired[solver_a], paired[solver_b])
            except ValueError:
                # every paired difference is exactly zero — wilcoxon can't be computed
                statistic, p_value = 0.0, 1.0
            rows.append(
                {
                    "solver_a": solver_a,
                    "solver_b": solver_b,
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "statistic": statistic,
                    "p_value": p_value,
                    "note": None,
                }
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "solver_a",
            "solver_b",
            "metric",
            "n_pairs",
            "statistic",
            "p_value",
            "note",
        ],
    )
    result["p_value_bh"] = np.nan
    computed = result["p_value"].notna()
    if computed.any():
        result.loc[computed, "p_value_bh"] = false_discovery_control(
            result.loc[computed, "p_value"].to_numpy(), method="bh"
        )
    return result


def aggregate_sweep(
    sweep_dir: str, output_dir: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """Runs the full aggregation pipeline and writes CSVs to
    <sweep_dir>/analysis/ (or output_dir if given). Returns the DataFrames
    too, so callers (CLI, future FastAPI endpoint) can use either the
    files or the in-memory result."""
    sweep_dir = Path(sweep_dir)
    out = Path(output_dir) if output_dir else sweep_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    runs_long = load_sweep(sweep_dir)
    if runs_long.empty:
        raise ValueError(f"No completed runs found in {sweep_dir}/index.json")

    runs_long = compute_optimality_gap(runs_long)
    summary_by_solver = compute_success_rate(runs_long).merge(
        compute_variable_reduction_stats(runs_long),
        on=["instance_map", "problem_name", "solver_name", "preprocess"],
    )
    statistical_tests = run_statistical_tests(runs_long)
    robot_statistics_long = load_robot_statistics(sweep_dir)

    runs_long.to_csv(out / "runs_long.csv", index=False)
    summary_by_solver.to_csv(out / "summary_by_solver.csv", index=False)
    statistical_tests.to_csv(out / "statistical_tests.csv", index=False)
    robot_statistics_long.to_csv(out / "robot_statistics_long.csv", index=False)

    return {
        "runs_long": runs_long,
        "summary_by_solver": summary_by_solver,
        "statistical_tests": statistical_tests,
        "robot_statistics_long": robot_statistics_long,
    }
