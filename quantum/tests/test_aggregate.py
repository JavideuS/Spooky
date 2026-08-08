"""
Phase 3 verification: aggregate_sweep()'s DataFrame schema and derived
columns (optimality_gap, success_rate) on a small, hand-built synthetic
sweep — not a real solve, just the on-disk shape load_sweep() expects
(index.json + benchmark_*.json), so this runs instantly and doesn't depend
on any solver actually being installed.
"""

import json

import pandas as pd
import pytest

from quantum.benchmark.analysis.aggregate import (
    compute_optimality_gap,
    compute_success_rate,
    load_robot_statistics,
    load_sweep,
    run_statistical_tests,
)


def _write_benchmark_json(path, problem_name, num_robots, runs):
    data = {
        "metadata": {
            "problem": {
                "name": problem_name,
                "robots": {f"r{i}": {} for i in range(num_robots)},
                "grid": {"M": 5, "N": 5, "obstacles": []},
            },
            "solver": {},
            "penalty_set": {},
            "benchmark_level": 2,
            "num_runs": len(runs),
            "timestamp": "2026-01-01T00:00:00",
        },
        "runs": runs,
        "summary": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


@pytest.fixture
def synthetic_sweep(tmp_path):
    sweep_dir = tmp_path / "sweep_test"
    sweep_dir.mkdir()

    def _robot_stats(efficiency):
        return {
            "total_robots": 1,
            "robot_statistics": {
                "r0": {
                    "path_length": 5, "moves_taken": 4, "optimal_path_length": 4.0,
                    "path_efficiency": efficiency, "goal_reached": True,
                    "validation_passed": True, "priority": 1.0,
                }
            },
            "successful_robots": 1, "success_rate": 1.0,
        }

    ilp_json = sweep_dir / "instA__p1__ilp_ref" / "benchmark_1.json"
    _write_benchmark_json(
        ilp_json, "instA_p1", num_robots=1,
        runs=[{"run_id": 1, "valid": True, "energy": 10.0, "execution_time_sec": 1.0,
               "termination_condition": "optimal", "solution_statistics": _robot_stats(1.0)}],
    )

    test_json = sweep_dir / "instA__p1__test_solver" / "benchmark_1.json"
    _write_benchmark_json(
        test_json, "instA_p1", num_robots=1,
        runs=[
            {"run_id": 1, "valid": True, "energy": 10.0, "execution_time_sec": 2.0,
             "solution_statistics": _robot_stats(0.8)},
            {"run_id": 2, "valid": True, "energy": 12.0, "execution_time_sec": 3.0,
             "solution_statistics": _robot_stats(0.6)},
            # run_id=3 deliberately has NO solution_statistics key, simulating a
            # BenchmarkRunner level<2 run — load_sweep/load_robot_statistics must
            # degrade gracefully (None columns / no robot rows), not KeyError.
            {"run_id": 3, "valid": False, "energy": 99.0, "execution_time_sec": 0.1},
        ],
    )

    index = [
        {"instance": "mapA", "problem": "p1", "solver": "ilp_ref", "backend": "ilp",
         "device": None, "penalty_set": None, "preprocess": True, "num_runs": 1,
         "output_dir": str(ilp_json.parent), "benchmark_json": str(ilp_json)},
        {"instance": "mapA", "problem": "p1", "solver": "test_solver", "backend": "dwave",
         "device": None, "penalty_set": "swap", "preprocess": True, "num_runs": 3,
         "output_dir": str(test_json.parent), "benchmark_json": str(test_json)},
    ]
    with open(sweep_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    return sweep_dir


def test_load_sweep_schema(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    assert len(df) == 4  # 1 ilp_ref run + 3 test_solver runs
    assert set(df["solver_name"]) == {"ilp_ref", "test_solver"}
    assert set(df.columns) >= {
        "instance_map", "problem_name", "num_robots", "solver_name", "backend",
        "preprocess", "valid", "energy", "execution_time_sec", "termination_condition",
    }
    assert (df["num_robots"] == 1).all()


def test_load_sweep_path_efficiency_columns(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    ilp_row = df[df["solver_name"] == "ilp_ref"].iloc[0]
    assert ilp_row["avg_path_efficiency"] == pytest.approx(1.0)
    assert ilp_row["min_path_efficiency"] == pytest.approx(1.0)
    assert ilp_row["robot_success_rate"] == pytest.approx(1.0)

    test_rows = df[df["solver_name"] == "test_solver"].sort_values("run_id")
    assert test_rows.iloc[0]["avg_path_efficiency"] == pytest.approx(0.8)
    assert test_rows.iloc[1]["avg_path_efficiency"] == pytest.approx(0.6)
    # run_id=3 has no solution_statistics at all -> NaN, not a crash/0.0
    assert pd.isna(test_rows.iloc[2]["avg_path_efficiency"])


def test_load_robot_statistics_schema_and_missing_data(synthetic_sweep):
    df = load_robot_statistics(synthetic_sweep)
    # 1 ilp_ref robot-run + 2 test_solver robot-runs (run_id=3 contributes none)
    assert len(df) == 3
    assert set(df.columns) >= {
        "instance_map", "problem_name", "solver_name", "run_id", "robot_id",
        "path_length", "path_efficiency", "validation_passed",
    }
    test_rows = df[df["solver_name"] == "test_solver"].sort_values("run_id")
    assert list(test_rows["path_efficiency"]) == pytest.approx([0.8, 0.6])
    assert set(test_rows["run_id"]) == {1, 2}  # confirms run_id=3 is absent


def test_optimality_gap_uses_only_proven_optimal_reference(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    df = compute_optimality_gap(df, reference_solver="ilp")

    ilp_row = df[df["solver_name"] == "ilp_ref"].iloc[0]
    assert ilp_row["reference_energy"] == 10.0
    assert ilp_row["optimality_gap"] == 0.0

    test_rows = df[df["solver_name"] == "test_solver"].sort_values("run_id")
    assert test_rows.iloc[0]["optimality_gap"] == 0.0  # energy 10 vs reference 10
    assert test_rows.iloc[1]["optimality_gap"] == pytest.approx(0.2)  # energy 12 vs reference 10
    # the invalid run (run_id=3) must not get a gap computed at all
    assert test_rows.iloc[2]["valid"] == False  # noqa: E712
    assert test_rows.iloc[2]["optimality_gap"] != test_rows.iloc[2]["optimality_gap"]  # NaN


def test_optimality_gap_missing_reference_is_nan_not_substituted(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    # No 'other_backend' rows with termination_condition == "optimal" exist at all
    df = compute_optimality_gap(df, reference_solver="other_backend")
    assert df["reference_missing"].all()
    assert df["optimality_gap"].isna().all()


def test_success_rate(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    summary = compute_success_rate(df)
    ilp_rate = summary[summary["solver_name"] == "ilp_ref"]["success_rate"].iloc[0]
    test_rate = summary[summary["solver_name"] == "test_solver"]["success_rate"].iloc[0]
    assert ilp_rate == 1.0
    assert test_rate == pytest.approx(2 / 3)


def test_statistical_tests_runs_without_error(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    df = compute_optimality_gap(df, reference_solver="ilp")
    result = run_statistical_tests(df, pairs=[("ilp_ref", "test_solver")])
    assert set(result["metric"]) == {"execution_time_sec", "optimality_gap"}


def test_statistical_tests_min_pairs_and_bh_correction():
    """Below _MIN_PAIRS_FOR_WILCOXON (6), no p-value is computed at all —
    the old `n_pairs < 1` guard let a 1-pair "result" through, which isn't
    a meaningful Wilcoxon test. At/above the threshold, a real p-value is
    computed and p_value_bh (Benjamini-Hochberg-adjusted, across the whole
    result set from this call) is populated alongside it."""
    rows = []
    for i in range(6):
        rows.append(
            {"instance_map": f"map{i}", "problem_name": "p", "solver_name": "solver_a",
             "valid": True, "execution_time_sec": 1.0, "optimality_gap": 0.0}
        )
        rows.append(
            {"instance_map": f"map{i}", "problem_name": "p", "solver_name": "solver_b",
             "valid": True, "execution_time_sec": 2.0, "optimality_gap": 0.1}
        )
    for i in range(2):  # only 2 shared instances with solver_a -- below threshold
        rows.append(
            {"instance_map": f"map{i}", "problem_name": "p", "solver_name": "solver_c",
             "valid": True, "execution_time_sec": 1.5, "optimality_gap": 0.05}
        )

    df = pd.DataFrame(rows)
    result = run_statistical_tests(
        df, pairs=[("solver_a", "solver_b"), ("solver_a", "solver_c")]
    )

    ab = result[(result["solver_a"] == "solver_a") & (result["solver_b"] == "solver_b")]
    ac = result[(result["solver_a"] == "solver_a") & (result["solver_b"] == "solver_c")]

    assert (ab["n_pairs"] == 6).all()
    assert ab["p_value"].notna().all()
    assert ab["p_value_bh"].notna().all()
    assert ab["note"].isna().all()

    assert (ac["n_pairs"] == 2).all()
    assert ac["p_value"].isna().all()
    assert ac["p_value_bh"].isna().all()
    assert ac["note"].str.contains("insufficient_data").all()
    assert (result["n_pairs"] >= 0).all()
