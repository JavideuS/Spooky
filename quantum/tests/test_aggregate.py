"""
Phase 3 verification: aggregate_sweep()'s DataFrame schema and derived
columns (energy_excess, success_rate) plus run_statistical_tests()'s pairing,
drop accounting and effect sizes, on a small, hand-built synthetic
sweep — not a real solve, just the on-disk shape load_sweep() expects
(index.json + benchmark_*.json), so this runs instantly and doesn't depend
on any solver actually being installed.
"""

import json

import numpy as np
import pandas as pd
import pytest

from quantum.benchmark.analysis.aggregate import (
    compute_energy_diagnostics,
    compute_energy_excess,
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


def test_energy_excess_is_relative_to_the_same_solvers_best_run(synthetic_sweep):
    """Energy is a within-solver quantity: a QUBO Hamiltonian value and an
    ILP objective are different things on different scales, so the reference
    is that configuration's own best valid run, never another solver's."""
    df = load_sweep(synthetic_sweep)
    df = compute_energy_excess(df)

    ilp_row = df[df["solver_name"] == "ilp_ref"].iloc[0]
    assert ilp_row["reference_energy"] == 10.0  # its own single run
    assert ilp_row["energy_excess"] == 0.0

    test_rows = df[df["solver_name"] == "test_solver"].sort_values("run_id")
    # best *valid* test_solver energy is 10.0 -- the invalid run's 99.0 must
    # not become the reference, and neither must ilp's identical 10.0
    assert (test_rows["reference_energy"].dropna() == 10.0).all()
    assert test_rows.iloc[0]["energy_excess"] == 0.0
    assert test_rows.iloc[1]["energy_excess"] == pytest.approx(2.0)  # 12.0 - 10.0
    assert test_rows.iloc[2]["valid"] == False  # noqa: E712
    assert pd.isna(test_rows.iloc[2]["energy_excess"])


def test_energy_excess_is_a_difference_not_a_ratio():
    """A QUBO Hamiltonian has no meaningful zero — apply_one_hot() drops the
    +K_hot constant of K*(sum(x)-1)^2 once per constraint — so a percentage
    of the reference is mostly percentage-of-offset. Adding a constant to
    every energy must leave the metric unchanged."""
    def _frame(offset):
        return pd.DataFrame([
            {"instance_map": "m", "problem_name": "p", "solver_name": "s",
             "penalty_set": "swap", "preprocess": True, "valid": True,
             "energy": -1.7667 + offset},
            {"instance_map": "m", "problem_name": "p", "solver_name": "s",
             "penalty_set": "swap", "preprocess": True, "valid": True,
             "energy": -1.0 + offset},
        ])

    # offset=0 is what the builder reports today; offset=108 is the same
    # solutions with the dropped one-hot constants (12 constraints * K_hot=9)
    dropped = compute_energy_excess(_frame(0.0))["energy_excess"].tolist()
    kept = compute_energy_excess(_frame(108.0))["energy_excess"].tolist()
    assert dropped == pytest.approx(kept)
    assert dropped == pytest.approx([0.0, 0.7667])

    # negative energies are the normal case and the worse (higher) run must
    # still get the positive excess
    assert dropped[1] > 0


def test_energy_excess_flags_inconsistent_window_counts():
    """energy is summed per window and each window drops its own constants,
    so runs that used different window counts are not comparable even within
    one configuration."""
    rows = [
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "penalty_set": "swap", "preprocess": True, "valid": True,
         "energy": -5.0, "num_windows": 1},
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "penalty_set": "swap", "preprocess": True, "valid": True,
         "energy": -4.0, "num_windows": 3},
    ]
    out = compute_energy_excess(pd.DataFrame(rows))
    assert out["energy_scale_mismatch"].all()

    rows[1]["num_windows"] = 1
    assert not compute_energy_excess(pd.DataFrame(rows))["energy_scale_mismatch"].any()


def test_energy_excess_groups_a_none_penalty_set_correctly():
    """penalty_set is routinely None; a groupby+merge on a NaN key never
    matches itself, which would leave every no-penalty-set row without a
    reference."""
    rows = [
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "penalty_set": None, "preprocess": True, "valid": True, "energy": 4.0},
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "penalty_set": None, "preprocess": True, "valid": True, "energy": 5.0},
    ]
    out = compute_energy_excess(pd.DataFrame(rows))
    assert not out["reference_missing"].any()
    assert out["energy_excess"].tolist() == [0.0, pytest.approx(1.0)]


def test_energy_excess_separates_different_penalty_sets():
    """Different penalty sets are different Hamiltonians -- their energies
    must not share a reference."""
    rows = [
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "penalty_set": "crash", "preprocess": True, "valid": True, "energy": 10.0},
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "penalty_set": "swap", "preprocess": True, "valid": True, "energy": 100.0},
    ]
    out = compute_energy_excess(pd.DataFrame(rows))
    assert out["reference_energy"].tolist() == [10.0, 100.0]
    assert out["energy_excess"].tolist() == [0.0, 0.0]


def test_compute_optimality_gap_is_refused(synthetic_sweep):
    """The old cross-solver energy comparison is gone rather than silently
    renamed -- a caller still using it must get an error, not a plausible
    number computed a different way."""
    df = load_sweep(synthetic_sweep)
    with pytest.raises(NotImplementedError, match="not a meaningful quantity"):
        compute_optimality_gap(df)


def test_valid_column_with_none_is_treated_as_false():
    """A run record missing the 'valid' key gives None, making the column
    object-dtype -- boolean indexing on that raises instead of treating the
    None as a failure."""
    rows = [
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "preprocess": True, "valid": True, "energy": 1.0},
        {"instance_map": "m", "problem_name": "p", "solver_name": "s",
         "preprocess": True, "valid": None, "energy": 2.0},
    ]
    df = pd.DataFrame(rows)
    assert df["valid"].dtype == object
    assert compute_success_rate(df)["success_rate"].iloc[0] == pytest.approx(0.5)
    assert pd.isna(compute_energy_excess(df)["energy_excess"].iloc[1])


def test_success_rate(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    summary = compute_success_rate(df)
    ilp_rate = summary[summary["solver_name"] == "ilp_ref"]["success_rate"].iloc[0]
    test_rate = summary[summary["solver_name"] == "test_solver"]["success_rate"].iloc[0]
    assert ilp_rate == 1.0
    assert test_rate == pytest.approx(2 / 3)


def test_statistical_tests_refuses_energy_metrics(synthetic_sweep):
    """Energy is not comparable across solvers, so a paired test on it is
    refused outright -- a computed p-value would be indistinguishable from
    a real one."""
    df = compute_energy_excess(load_sweep(synthetic_sweep))
    result = run_statistical_tests(
        df, pairs=[("test_solver", "ilp_ref")], metrics=("energy_excess", "energy")
    )
    assert len(result) == 2
    assert result["p_value"].isna().all()
    assert result["note"].str.startswith("not_comparable_across_solvers").all()


def _paired_frame(spec, metric="execution_time_sec"):
    """spec: {solver_name: [per-instance value or None]} -- None means the
    solver didn't attempt that instance at all."""
    rows = []
    for solver, values in spec.items():
        for i, value in enumerate(values):
            if value is None:
                continue
            valid = value is not False
            rows.append({
                "instance_map": f"map{i}", "problem_name": "p",
                "solver_name": solver, "backend": solver, "preprocess": True,
                "valid": valid, metric: (value if valid else None),
                "termination_condition": None,
            })
    return pd.DataFrame(rows)


def test_statistical_tests_report_direction_and_effect_size():
    """A p-value with no sign and no effect size cannot say who won:
    scipy's two-sided statistic is min(W+, W-), which carries no direction."""
    n = 8
    df = _paired_frame({
        "fast": [1.0] * n,
        "slow": [4.0] * (n - 1) + [8.0],  # never ties, so no zero differences
    })
    result = run_statistical_tests(df, pairs=[("fast", "slow")],
                                   metrics=("execution_time_sec",))
    row = result.iloc[0]

    assert row["favors"] == "fast"                    # lower time is better
    assert row["rank_biserial"] == pytest.approx(-1.0)  # fast is smaller everywhere
    assert row["transform"] == "log"
    assert row["median_ratio"] == pytest.approx(0.25)   # fast takes 1/4 the time
    assert row["median_diff"] == pytest.approx(np.log(0.25))
    assert row["p_value"] is not None and row["p_value"] < 0.05


def test_statistical_tests_favors_flips_for_higher_is_better_metrics():
    n = 8
    df = _paired_frame({"good": [0.9] * n, "bad": [0.5] * n},
                       metric="avg_path_efficiency")
    row = run_statistical_tests(df, pairs=[("good", "bad")],
                                metrics=("avg_path_efficiency",)).iloc[0]
    assert row["favors"] == "good"
    assert row["rank_biserial"] == pytest.approx(1.0)
    assert pd.isna(row["median_ratio"])  # untransformed metric


def test_statistical_tests_uses_log_scale_for_time():
    """Wilcoxon ranks absolute differences, so on the raw scale the one huge
    instance outranks every small one. On the log scale a consistent 2x
    speedup on seven small instances outweighs one 5% loss on a big one."""
    df = _paired_frame({
        "cand": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 630.0],
        "base": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 600.0],
    })
    row = run_statistical_tests(df, pairs=[("cand", "base")],
                                metrics=("execution_time_sec",)).iloc[0]
    assert row["favors"] == "cand"
    # on raw differences the +30s instance has by far the largest |diff| and
    # would take rank 8; on the log scale its 1.05x ratio ranks last
    raw_ranks_top = np.argmax(np.abs(np.array([1.0] * 7 + [630.0])
                                     - np.array([2.0] * 7 + [600.0])))
    assert raw_ranks_top == 7
    assert row["median_ratio"] == pytest.approx(0.5)


def test_statistical_tests_report_effective_n_separately_from_n_pairs():
    """zero_method='wilcox' discards zero differences, so the test's real n
    is smaller than the number of pairs whenever the solvers tie."""
    df = _paired_frame({
        "cand": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "base": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    })
    row = run_statistical_tests(df, pairs=[("cand", "base")],
                                metrics=("execution_time_sec",)).iloc[0]
    assert row["n_pairs"] == 10
    assert row["n_effective"] == 3
    # 3 effective pairs is below the threshold: no p-value is reported even
    # though n_pairs alone would have passed the guard
    assert pd.isna(row["p_value"])
    assert "insufficient_effective_pairs" in row["note"]
    # ...but the effect size is still there
    assert row["favors"] == "cand"


def test_statistical_tests_account_for_dropped_instances():
    """The paired population is conditioned on both solvers producing a
    usable value, so it shifts per pair. Make the shift visible instead of
    silent: unshared, invalid and metric-missing drops are counted apart."""
    df = _paired_frame({
        # 6 shared+valid, 1 instance cand never attempted, 1 where it failed
        "cand": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None, False],
        "base": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    })
    row = run_statistical_tests(df, pairs=[("cand", "base")],
                                metrics=("execution_time_sec",)).iloc[0]
    assert row["n_pairs"] == 6
    assert row["n_dropped_unshared"] == 1
    assert row["n_dropped_invalid"] == 1
    assert row["n_dropped_no_metric"] == 0


def test_statistical_tests_flag_metric_recorded_but_missing():
    """A run can be valid and still have no value for the metric (path
    efficiency needs BenchmarkRunner level>=2) -- that is a different kind
    of drop from an invalid run."""
    df = _paired_frame({"cand": [0.9] * 7, "base": [0.5] * 7},
                       metric="avg_path_efficiency")
    df.loc[df.index[-1], "avg_path_efficiency"] = np.nan  # valid, no metric
    row = run_statistical_tests(df, pairs=[("cand", "base")],
                                metrics=("avg_path_efficiency",)).iloc[0]
    assert row["n_pairs"] == 6
    assert row["n_dropped_invalid"] == 0
    assert row["n_dropped_no_metric"] == 1


def test_statistical_tests_expose_thin_per_instance_support():
    """A per-instance value is the mean over that solver's *valid* runs, so
    one lucky run out of twenty can represent an instance. min_valid_runs
    makes that visible."""
    rows = []
    for i in range(6):
        for run_id in range(4):
            # cand is valid on only 1 of its 4 runs on map0
            valid = not (i == 0 and run_id < 3)
            rows.append({"instance_map": f"map{i}", "problem_name": "p",
                         "solver_name": "cand", "backend": "cand", "preprocess": True,
                         "valid": valid, "execution_time_sec": 1.0 if valid else None,
                         "termination_condition": None})
            rows.append({"instance_map": f"map{i}", "problem_name": "p",
                         "solver_name": "base", "backend": "base", "preprocess": True,
                         "valid": True, "execution_time_sec": 2.0,
                         "termination_condition": None})
    row = run_statistical_tests(pd.DataFrame(rows), pairs=[("cand", "base")],
                                metrics=("execution_time_sec",)).iloc[0]
    assert row["n_pairs"] == 6
    assert row["min_valid_runs"] == 1


def test_statistical_tests_flag_censored_times():
    """A run cut off at a time/node limit reports a lower bound, not a
    duration -- Wilcoxon over those understates the true difference."""
    df = _paired_frame({"cand": [1.0] * 7, "base": [2.0] * 7})
    df.loc[
        (df["solver_name"] == "base") & (df["instance_map"] == "map0"),
        "termination_condition",
    ] = "time_limit_exceeded"
    row = run_statistical_tests(df, pairs=[("cand", "base")],
                                metrics=("execution_time_sec",)).iloc[0]
    assert row["n_censored"] == 1
    assert "censored lower bounds" in row["note"]


def test_statistical_tests_flag_non_positive_times_instead_of_crashing():
    df = _paired_frame({"cand": [1.0] * 7, "base": [2.0] * 6 + [0.0]})
    row = run_statistical_tests(df, pairs=[("cand", "base")],
                                metrics=("execution_time_sec",)).iloc[0]
    assert pd.isna(row["p_value"])
    assert "log_transform_invalid" in row["note"]


def test_default_pairs_put_the_baseline_second():
    """Direction has to survive pair construction: sorting each pair
    alphabetically loses which solver is the baseline."""
    from quantum.benchmark.analysis.aggregate import _default_pairs

    df = pd.DataFrame([
        {"solver_name": "aaa_quantum", "backend": "dwave"},
        {"solver_name": "zzz_quantum", "backend": "pennylane"},
        {"solver_name": "ilp_ref", "backend": "ilp"},
    ])
    pairs = _default_pairs(df)
    assert set(pairs) == {("aaa_quantum", "ilp_ref"), ("zzz_quantum", "ilp_ref")}


def test_statistical_tests_runs_without_error(synthetic_sweep):
    df = load_sweep(synthetic_sweep)
    df = compute_energy_excess(df)
    result = run_statistical_tests(df, pairs=[("test_solver", "ilp_ref")])
    assert set(result["metric"]) == {"execution_time_sec", "avg_path_efficiency"}


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
             "valid": True, "execution_time_sec": 1.0, "avg_path_efficiency": 0.9}
        )
        rows.append(
            {"instance_map": f"map{i}", "problem_name": "p", "solver_name": "solver_b",
             "valid": True, "execution_time_sec": 2.0, "avg_path_efficiency": 0.5}
        )
    for i in range(2):  # only 2 shared instances with solver_a -- below threshold
        rows.append(
            {"instance_map": f"map{i}", "problem_name": "p", "solver_name": "solver_c",
             "valid": True, "execution_time_sec": 1.5, "avg_path_efficiency": 0.7}
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


def _energy_frame(runs):
    """runs: list of (energy, valid, avg_path_efficiency)."""
    return pd.DataFrame([
        {"instance_map": "m", "problem_name": "p", "solver_name": "qaoa",
         "penalty_set": "swap", "preprocess": True, "num_windows": 1,
         "energy": e, "valid": v, "avg_path_efficiency": eff}
        for e, v, eff in runs
    ])


def test_energy_diagnostics_summarize_stochastic_spread():
    """For a stochastic backend one run's energy says almost nothing — the
    distribution over repeated runs is the metric."""
    df = _energy_frame([(-10.0, True, 1.0), (-10.0, True, 1.0),
                        (-8.0, True, 0.8), (-6.0, True, 0.6)])
    row = compute_energy_diagnostics(df).iloc[0]
    assert row["n_runs"] == 4 and row["n_energy_runs"] == 4
    assert row["best_energy"] == -10.0
    assert row["hit_rate_best"] == pytest.approx(0.5)   # 2 of 4 runs reach it
    assert row["median_excess"] == pytest.approx(1.0)
    assert row["max_excess"] == pytest.approx(4.0)
    assert row["windows_consistent"]


def test_energy_diagnostics_detect_penalty_too_weak():
    """The sharp formulation test: an invalid bitstring scoring below a valid
    one is a counterexample, not a judgment call."""
    df = _energy_frame([(-5.0, True, 1.0), (-4.0, True, 0.9),
                        (-9.0, False, None)])  # invalid run has the LOWEST energy
    row = compute_energy_diagnostics(df).iloc[0]
    assert not row["separated"]  # pandas re-boxes the column as np.bool_
    assert row["n_inversions"] == 2          # below both valid runs
    assert row["separation_auc"] == 0.0
    assert "too weak" in row["note"]


def test_energy_diagnostics_clean_separation():
    df = _energy_frame([(-9.0, True, 1.0), (-8.0, True, 0.9),
                        (-2.0, False, None), (-1.0, False, None)])
    row = compute_energy_diagnostics(df).iloc[0]
    assert row["separated"]
    assert row["n_inversions"] == 0
    assert row["separation_auc"] == 1.0


def test_energy_diagnostics_rank_correlate_energy_with_quality():
    """A well-balanced QUBO gives a negative rho: lower energy is better,
    higher efficiency is better."""
    df = _energy_frame([(-10.0, True, 1.0), (-8.0, True, 0.8),
                        (-6.0, True, 0.6), (-4.0, True, 0.4)])
    row = compute_energy_diagnostics(df).iloc[0]
    assert row["quality_n"] == 4
    assert row["quality_rho"] == pytest.approx(-1.0)
    assert row["note"] is None


def test_energy_diagnostics_flag_inverted_quality_correlation():
    """rho > 0 means lower energy is picking the *worse* paths — the penalty
    set is miscalibrated."""
    df = _energy_frame([(-10.0, True, 0.4), (-8.0, True, 0.6),
                        (-6.0, True, 0.8), (-4.0, True, 1.0)])
    row = compute_energy_diagnostics(df).iloc[0]
    assert row["quality_rho"] == pytest.approx(1.0)
    assert "selecting *worse* paths" in row["note"]


def test_energy_diagnostics_rank_correlation_is_offset_invariant():
    """Rank-based on purpose: immune to the dropped additive constant and to
    normalize_scale, which both destroy the magnitude but not the ordering."""
    runs = [(-10.0, True, 1.0), (-8.0, True, 0.8), (-6.0, True, 0.6)]
    plain = compute_energy_diagnostics(_energy_frame(runs)).iloc[0]["quality_rho"]
    shifted = compute_energy_diagnostics(
        _energy_frame([(e + 108.0, v, eff) for e, v, eff in runs])
    ).iloc[0]["quality_rho"]
    scaled = compute_energy_diagnostics(
        _energy_frame([(e * 4.0, v, eff) for e, v, eff in runs])
    ).iloc[0]["quality_rho"]
    assert plain == pytest.approx(shifted) == pytest.approx(scaled)


def test_energy_diagnostics_degenerate_cases_do_not_crash():
    """Constant energies (deterministic solver) and too-few runs must give
    NaN, not a warning-laden bogus correlation."""
    constant = compute_energy_diagnostics(
        _energy_frame([(-5.0, True, 1.0)] * 4)
    ).iloc[0]
    assert pd.isna(constant["quality_rho"])
    assert constant["hit_rate_best"] == 1.0

    all_invalid = compute_energy_diagnostics(
        _energy_frame([(-5.0, False, None), (-4.0, False, None)])
    ).iloc[0]
    assert all_invalid["n_energy_runs"] == 0
    assert pd.isna(all_invalid["best_energy"])
    assert pd.isna(all_invalid["separation_auc"])
    assert "no_energy_runs" in all_invalid["note"]


def test_energy_diagnostics_one_row_per_configuration():
    df = pd.concat([
        _energy_frame([(-5.0, True, 1.0), (-4.0, True, 0.9)]),
        _energy_frame([(-50.0, True, 1.0)]).assign(penalty_set="crash"),
    ], ignore_index=True)
    out = compute_energy_diagnostics(df)
    assert len(out) == 2
    assert set(out["penalty_set"]) == {"swap", "crash"}
    assert sorted(out["best_energy"]) == [-50.0, -5.0]


def _skipped_frame(runs):
    """runs: list of (energy, valid, total_final_variables)."""
    return pd.DataFrame([
        {"instance_map": "m", "problem_name": "p", "solver_name": "qaoa",
         "penalty_set": "crash", "preprocess": True, "num_windows": 1,
         "energy": e, "valid": v, "total_final_variables": fv,
         "average_reduction_ratio": 1.0 if fv == 0 else 0.5,
         "avg_path_efficiency": None}
        for e, v, fv in runs
    ])


def test_energy_excess_ignores_runs_where_the_solver_never_ran():
    """solve_qubo_smart's BFS can consume every variable ("Window N: 31 -> 0
    vars, skipping solver"). The recorded energy is then 0.0, a sentinel. On
    a *valid* run that 0.0 would be the smallest number in its class and
    would become the reference, poisoning every other run's excess."""
    df = _skipped_frame([(-10.0, True, 7), (-8.0, True, 7), (0.0, True, 0)])
    out = compute_energy_excess(df)

    assert (out["reference_energy"].dropna() == -10.0).all()  # not 0.0
    assert out["energy_excess"].iloc[0] == 0.0
    assert out["energy_excess"].iloc[1] == pytest.approx(2.0)
    assert pd.isna(out["energy_excess"].iloc[2])              # the skipped run


def test_energy_diagnostics_report_skipped_and_preprocessing_share():
    df = _skipped_frame([(-10.0, True, 7), (0.0, False, 0), (0.0, False, 0)])
    row = compute_energy_diagnostics(df).iloc[0]
    assert row["n_solver_skipped"] == 2
    assert row["n_energy_runs"] == 1
    assert row["best_energy"] == -10.0    # the sentinels are excluded
    assert "sentinel" in row["note"]


def test_energy_diagnostics_warn_when_preprocessing_did_the_work():
    """A run that is 97% pre-processed is measuring BFS, not the solver."""
    df = pd.DataFrame([
        {"instance_map": "m", "problem_name": "p", "solver_name": "qaoa",
         "penalty_set": "crash", "preprocess": True, "num_windows": 9,
         "energy": -2.6, "valid": True, "total_final_variables": 8,
         "average_reduction_ratio": 0.974, "avg_path_efficiency": 0.7}
    ])
    row = compute_energy_diagnostics(df).iloc[0]
    assert row["median_reduction_ratio"] == pytest.approx(0.974)
    assert "mostly measuring BFS" in row["note"]


def test_exact_solvers_are_not_treated_as_skipped():
    """ILP/CBS have no variable_stats at all — a missing total_final_variables
    means "ran", not "skipped"."""
    df = pd.DataFrame([
        {"instance_map": "m", "problem_name": "p", "solver_name": "ilp",
         "penalty_set": None, "preprocess": True, "energy": 4.0, "valid": True},
        {"instance_map": "m", "problem_name": "p", "solver_name": "ilp",
         "penalty_set": None, "preprocess": True, "energy": 6.0, "valid": True},
    ])
    out = compute_energy_excess(df)
    assert out["energy_excess"].tolist() == [0.0, pytest.approx(2.0)]
    assert compute_energy_diagnostics(df).iloc[0]["n_solver_skipped"] == 0
