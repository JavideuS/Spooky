"""
The build-only QUBO size measurement in
quantum.benchmark.analysis.variable_scaling.

These build real QUBOs against the committed 3x3 maps but never invoke a
solver, so they stay fast and cannot fail on qubit limits -- which is the
whole reason the module exists.
"""

import pandas as pd
import pytest

from quantum.benchmark.analysis.variable_scaling import (
    instances_from_sweep_config,
    measure_instance,
    measure_scaling,
    relative_to_naive,
    relative_to_raw,
)
from quantum.utils import preprocess as pm

MAP = "quantum/maps/synthetic/3x3/no_obs3x3"
PENALTIES = {
    "K_hot": 9, "K_adj": 4.8, "K_start": 6.5, "K_goal": 3, "K_lock": 4,
    "K_bt": 2.3, "K_tp": 1.2, "K_goal_approx": 0.7, "K_obs": 0,
    "K_crash": 2.7, "K_swap": 3,
}


@pytest.fixture(scope="module")
def rows():
    return {
        r["preprocess"]: r
        for r in measure_instance(MAP, "baseline", PENALTIES, pm.MODES)
    }


def test_every_mode_is_measurable_without_a_solver(rows):
    assert set(rows) == set(pm.MODES)
    assert all(r["error"] is None for r in rows.values())


def test_raw_is_the_unpruned_baseline(rows):
    """raw is the mode a sweep is least likely to record, and the one every
    other mode's reduction ratio is implicitly measured against."""
    raw = rows[pm.RAW]["window_variables"]
    for mode in (pm.BFS_AGGRESSIVE, pm.BFS_SAFE, pm.FULL, pm.FULL_SAFE):
        assert rows[mode]["window_variables"] <= raw, mode


def test_safe_bfs_prunes_less_than_aggressive(rows):
    """The monotone set allows waiting and revisiting, so it can only be a
    superset of the frontier -- if this ever inverts, the safe variant is not
    actually safe."""
    assert (
        rows[pm.BFS_SAFE]["window_variables"]
        > rows[pm.BFS_AGGRESSIVE]["window_variables"]
    )


def test_numeric_modes_only_differ_after_the_numeric_stage(rows):
    """full shares bfs_aggressive's BFS and full_safe shares bfs_safe's, so
    the pre-numeric counts must match exactly; only the post-numeric count
    may drop."""
    assert (
        rows[pm.FULL]["window_variables"]
        == rows[pm.BFS_AGGRESSIVE]["window_variables"]
    )
    assert (
        rows[pm.FULL_SAFE]["window_variables"] == rows[pm.BFS_SAFE]["window_variables"]
    )
    assert (
        rows[pm.FULL]["window_variables_after_numeric"]
        < rows[pm.FULL]["window_variables"]
    )
    # modes without the numeric stage must not shrink at all
    for mode in (pm.BFS_AGGRESSIVE, pm.BFS_SAFE):
        assert (
            rows[mode]["window_variables_after_numeric"]
            == rows[mode]["window_variables"]
        )


def test_encoded_variables_is_mode_independent(rows):
    """Pruning removes variables from the window, not from the problem, so the
    full encoding size is the denominator that makes instances comparable."""
    assert len({r["encoded_variables"] for r in rows.values()}) == 1


def test_a_malformed_problem_is_reported_not_raised():
    """obs5x5_hard/three_robots puts robot_2's goal on an obstacle. One bad
    instance must not abort a scan of thirty good ones."""
    out = measure_instance(
        "quantum/maps/synthetic/5x5/obs5x5_hard",
        "three_robots",
        PENALTIES,
        (pm.BFS_AGGRESSIVE,),
    )
    assert len(out) == 1
    assert "on an obstacle" in out[0]["error"]


def test_relative_to_raw_normalizes_per_instance():
    df = relative_to_raw(
        measure_scaling([(MAP, "baseline")], PENALTIES, (pm.RAW, pm.BFS_AGGRESSIVE))
    )
    raw = df[df["preprocess"] == pm.RAW].iloc[0]
    aggressive = df[df["preprocess"] == pm.BFS_AGGRESSIVE].iloc[0]
    assert raw["vs_raw"] == pytest.approx(1.0)
    assert 0 < aggressive["vs_raw"] < 1


def test_instances_come_from_the_sweep_config_verbatim():
    """The scaling table must cover exactly what the sweep runs, so the two
    outputs join on (instance_map, problem_name, preprocess)."""
    config = {
        "instances": [
            {"map": "a", "problems": ["p1", "p2"]},
            {"map": "b", "problems": ["p3"]},
        ]
    }
    assert instances_from_sweep_config(config) == [("a", "p1"), ("a", "p2"), ("b", "p3")]
    assert instances_from_sweep_config({}) == []


def test_raw_is_the_unpruned_window_not_the_whole_problem():
    """The distinction the columns exist to preserve: windowing cuts the
    encoding before any BFS runs, so attributing raw's whole gap to
    pre-processing would overstate what BFS does.

    5x5/two_robots is used rather than the 3x3 fixture because on the 3x3 the
    window (13 steps) already spans the whole horizon (12), so windowing
    reduces nothing there and raw == encoded. That is the honest edge case:
    windowing only pays off once the horizon outgrows one window.
    """
    rows = {
        r["preprocess"]: r
        for r in measure_instance(
            "quantum/maps/synthetic/5x5/obs5x5_easy",
            "two_robots",
            PENALTIES,
            (pm.RAW,),
        )
    }
    raw = rows[pm.RAW]
    assert raw["num_windows"] > 1
    assert raw["window_variables"] < raw["encoded_variables"]
    assert raw["fraction_of_encoding"] == pytest.approx(
        raw["window_variables"] / raw["encoded_variables"]
    )


def test_windowing_is_identical_across_modes(rows):
    """Modes differ only in pruning, never in how the horizon is cut, so
    vs_raw isolates pruning cleanly."""
    assert len({r["window_max_steps"] for r in rows.values()}) == 1
    assert len({r["num_windows"] for r in rows.values()}) == 1


def test_a_single_window_horizon_reduces_nothing(rows):
    """On the 3x3 the window spans the whole horizon, so raw's window is the
    full encoding -- windowing is not a free reduction, it only helps once
    the horizon exceeds one window."""
    raw = rows[pm.RAW]
    assert raw["num_windows"] == 1
    assert raw["window_variables"] == raw["encoded_variables"]


def test_peak_and_total_are_reported_separately(rows):
    """window_variables is what a qubit budget pays for (one window held at a
    time); total_window_variables is what compute pays for. Windowing pushes
    them in opposite directions, so reporting only one is misleading."""
    for mode, row in rows.items():
        assert row["num_windows"] >= 1, mode
        assert row["total_window_variables"] == (
            row["window_variables_after_numeric"] * row["num_windows"]
        ), mode


def test_windowing_gain_depends_on_the_pruning():
    """Windowing is not a free reduction and not a pure concession either.

    With no pruning it costs MORE in total, because consecutive windows
    overlap by a timestep and re-cover the same cells. With sound pruning it
    wins by a wide margin, because a monotone reachable set saturates to the
    whole free space over a long horizon and stays there, while windowing
    re-seeds it small every couple of steps and keeps BFS in the early-steps
    regime where it prunes hard.
    """
    rows = {
        r["preprocess"]: r
        for r in measure_instance(
            "quantum/maps/synthetic/5x5/obs5x5_easy",
            "two_robots",
            PENALTIES,
            (pm.RAW, pm.BFS_SAFE),
        )
    }
    raw, safe = rows[pm.RAW], rows[pm.BFS_SAFE]
    assert raw["num_windows"] > 1

    # unpruned: the overlap is pure overhead
    assert raw["total_window_variables"] > raw["full_horizon_variables"]
    assert raw["windowing_gain"] < 1

    # sound pruning: windowing more than pays for the overlap
    assert safe["total_window_variables"] < safe["full_horizon_variables"]
    assert safe["windowing_gain"] > 1


def test_full_horizon_can_be_skipped():
    """It is the only measurement that scales with the map, so a very large
    grid must be measurable without it."""
    rows = measure_instance(
        MAP, "baseline", PENALTIES, (pm.BFS_SAFE,), measure_full_horizon=False
    )
    assert rows[0]["full_horizon_variables"] is None
    assert rows[0]["windowing_gain"] is None
    assert rows[0]["window_variables"] > 0


def test_raw_windowing_never_beats_the_naive_baseline():
    """The demonstration the naive_gain column exists for.

    Windowing with no pruning can only lose: consecutive windows overlap by a
    timestep and re-cover the same cells, and nothing is removed to offset it.
    The best raw can do is break even, on an instance whose horizon fits in a
    single window -- and then it is not really windowed at all.
    """
    instances = [
        ("quantum/maps/synthetic/3x3/no_obs3x3", "baseline"),          # 1 window
        ("quantum/maps/synthetic/5x5/obs5x5_easy", "two_robots"),      # 4 windows
        ("quantum/maps/synthetic/5x5/obs5x5_easy", "three_robots"),    # 6 windows
    ]
    df = relative_to_naive(
        measure_scaling(instances, PENALTIES, (pm.RAW, pm.BFS_SAFE))
    )

    raw = df[df["preprocess"] == pm.RAW]
    assert (raw["naive_gain"] <= 1.0 + 1e-9).all(), raw[
        ["instance_map", "problem_name", "naive_gain"]
    ]
    # break-even only where there is a single window; a real split costs more
    multi = raw[raw["num_windows"] > 1]
    assert not multi.empty
    assert (multi["naive_gain"] < 1.0).all()

    # pruning is what turns windowing from a cost into a saving
    safe = df[df["preprocess"] == pm.BFS_SAFE]
    assert (safe["naive_gain"] > 1.0).all()


def test_naive_gain_is_measured_against_raw_full_horizon():
    """Not against raw's windowed count: the baseline is 'write the problem
    down and hand it over', which is one un-windowed QUBO with no pruning."""
    df = relative_to_naive(
        measure_scaling(
            [("quantum/maps/synthetic/5x5/obs5x5_easy", "two_robots")],
            PENALTIES,
            (pm.RAW, pm.BFS_SAFE),
        )
    )
    naive = df[df["preprocess"] == pm.RAW].iloc[0]["full_horizon_variables"]
    safe = df[df["preprocess"] == pm.BFS_SAFE].iloc[0]
    assert safe["naive_gain"] == pytest.approx(
        naive / safe["total_window_variables"]
    )


def test_an_oversized_window_is_skipped_not_built():
    """`raw` at 50x50 is cells x robots x t_max -- 30,000 variables for four
    robots, and the QUBO build is quadratic in pairwise terms. Building it
    stalled the whole table with no output; it is now reported as a skip."""
    rows = measure_instance(
        "quantum/maps/synthetic/50x50/no_obs50x50",
        "four_robots",
        PENALTIES,
        (pm.RAW, pm.BFS_AGGRESSIVE),
        measure_full_horizon=False,
    )
    by_mode = {r["preprocess"]: r for r in rows}
    assert "exceeds" in by_mode[pm.RAW]["error"]
    # the pruned mode on the same instance stays tiny and is measured
    assert by_mode[pm.BFS_AGGRESSIVE]["error"] is None
    assert by_mode[pm.BFS_AGGRESSIVE]["window_variables"] < 100


def test_error_rows_carry_the_full_schema():
    """relative_to_raw() reads window_variables. A narrower error row made it
    KeyError as soon as every mode for an instance failed -- which is exactly
    what a large map produces."""
    rows = measure_instance(
        "quantum/maps/synthetic/5x5/obs5x5_hard",  # malformed: goal on obstacle
        "three_robots",
        PENALTIES,
        (pm.RAW, pm.BFS_SAFE),
    )
    assert all(r["error"] is not None for r in rows)
    df = relative_to_naive(relative_to_raw(pd.DataFrame(rows)))
    assert "window_variables" in df.columns
    assert df["window_variables"].isna().all()


def test_sweep_dir_addressing_matches_the_other_analysis_clis():
    """run_aggregate/run_plots/run_report all take --sweep-dir; this one took
    only --config or --map, so pointing it at a finished sweep meant
    remembering which config produced it."""
    from quantum.benchmark.analysis.variable_scaling import (
        penalty_set_from_config,
    )

    config = {
        "instances": [{"map": "m", "problems": ["p1", "p2"]}],
        "solvers": [
            {"name": "ilp", "penalty_set": None},
            {"name": "sa", "penalty_set": "crash"},
            {"name": "qaoa", "penalty_set": "crash"},
        ],
    }
    assert instances_from_sweep_config(config) == [("m", "p1"), ("m", "p2")]
    # ILP/CBS record None and must not confuse the majority
    assert penalty_set_from_config(config) == "crash"
    # genuinely ambiguous configs get no default rather than a wrong one
    config["solvers"][1]["penalty_set"] = "swap"
    assert penalty_set_from_config(config) is None
