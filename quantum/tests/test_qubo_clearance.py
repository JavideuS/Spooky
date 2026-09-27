"""
Robot-robot clearance in QUBO (quantum/docs/clearance.md): apply_crash_penalty
and apply_trailing_penalty generalise their same-cell matching to the D_ab
shell, the same idea already verified for CBS/ILP in test_cbs.py.

Structural checks on the Q-matrix are the primary evidence here — they're
fast and deterministic, unlike a simulated-annealing solve. One end-to-end
DWave (classical simulated annealing) solve is included for genuine
confidence that a decoded solution actually respects separation.
"""

import math

from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem
from quantum.builder.QUBOBuilder import GridQUBOBuilder
from quantum.solvers.DWave_solver import DWaveSolver
from quantum.benchmark.benchmark import is_solution_valid, _attribute_invalid_cause

PENALTIES = {
    "K_hot": 9, "K_adj": 4.8, "K_start": 6.5, "K_goal": 3, "K_lock": 4,
    "K_bt": 2.3, "K_tp": 1.2, "K_goal_approx": 0.7, "K_obs": 0,
    "K_crash": 2.7, "K_trail": 3,
}


def _cheb(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def test_no_clearance_terms_at_defaults():
    """Default robot_radius -> D_ab = {(0,0)} for every pair -> every crash/
    trailing Q entry pairs the SAME cell for both robots. Byte-for-byte the
    pre-clearance model."""
    grid = Grid(5, 5, obstacles=[])
    robots = [
        RobotConfig("a", (0, 0), (4, 4), start_time=0, expected_duration=10),
        RobotConfig("b", (0, 4), (4, 0), start_time=0, expected_duration=10),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="qubo_default")
    builder = GridQUBOBuilder(problem, penalties=PENALTIES)
    builder.build()

    N = grid.N
    M = grid.M
    robot_nums = problem.get_robot_nums()
    total_t = builder.total_t
    offset_a = robot_nums["a"] * (M * N * total_t)
    offset_b = robot_nums["b"] * (M * N * total_t)

    def decode(idx, offset):
        local = idx - offset
        t = local // (M * N)
        rem = local % (M * N)
        i, j = divmod(rem, N)
        return i, j, t

    mismatches = []
    for (idx1, idx2), weight in builder.Q.items():
        if weight not in (PENALTIES["K_crash"], PENALTIES["K_trail"]):
            continue
        if not (offset_a <= idx1 < offset_a + M * N * total_t):
            continue
        if not (offset_b <= idx2 < offset_b + M * N * total_t):
            continue
        i1, j1, t1 = decode(idx1, offset_a)
        i2, j2, t2 = decode(idx2, offset_b)
        if (i1, j1) != (i2, j2):
            mismatches.append(((i1, j1, t1), (i2, j2, t2)))

    assert mismatches == [], f"clearance-shell terms found at default radius: {mismatches[:5]}"


def test_crash_and_trailing_generalise_to_dab_shell():
    """A real robot_radius must add cross-cell Q terms (different cells,
    within the D_ab shell) beyond the same-cell ones."""
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=16,
                    robot_radius=0.6),
        RobotConfig("b", (3, 3), (3, 3), start_time=0, expected_duration=16,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="qubo_clearance")
    D = problem.get_clearance_table()[("a", "b")]
    assert D != frozenset({(0, 0)})  # sanity: this pair is a clearance pair

    builder = GridQUBOBuilder(problem, penalties=PENALTIES)
    builder.build()

    N, M = grid.N, grid.M
    robot_nums = problem.get_robot_nums()
    total_t = builder.total_t
    offset_a = robot_nums["a"] * (M * N * total_t)
    offset_b = robot_nums["b"] * (M * N * total_t)

    def decode(idx, offset):
        local = idx - offset
        t = local // (M * N)
        rem = local % (M * N)
        i, j = divmod(rem, N)
        return i, j, t

    cross_cell_terms = 0
    for (idx1, idx2), weight in builder.Q.items():
        if weight not in (PENALTIES["K_crash"], PENALTIES["K_trail"]):
            continue
        if not (offset_a <= idx1 < offset_a + M * N * total_t):
            continue
        if not (offset_b <= idx2 < offset_b + M * N * total_t):
            continue
        i1, j1, t1 = decode(idx1, offset_a)
        i2, j2, t2 = decode(idx2, offset_b)
        if (i1, j1) != (i2, j2):
            assert _cheb((i1, j1), (i2, j2)) <= max(_cheb((0, 0), d) for d in D), (
                f"cross-cell term ({i1},{j1}) vs ({i2},{j2}) falls outside D_ab"
            )
            cross_cell_terms += 1

    assert cross_cell_terms > 0, "expected D_ab-shell cross-cell terms for a clearance pair"


def test_dwave_solve_respects_clearance():
    """End-to-end: classical simulated annealing (no cloud token, see
    DWave_solver.py) on a clearance-configured crossing instance decodes to a
    solution is_solution_valid() accepts, including its clearance/trailing
    checks. Window 0 resolves via BFS preprocessing alone (the two shortest
    paths happen not to conflict); window 1 genuinely reaches the annealer
    for robot b's remaining steps -- so this exercises real solving, not
    just preprocessing, without hitting the gap below.

    The preprocessing side of this (BFS/diagonal fixing committing a
    clearance violation on its own) is covered by
    test_preprocessing_routes_around_clearance_shell below."""
    grid = Grid(5, 5, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (2, 0), (2, 4), start_time=0, expected_duration=10,
                    robot_radius=0.4),
        RobotConfig("b", (0, 2), (4, 2), start_time=0, expected_duration=10,
                    robot_radius=0.4),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="qubo_dwave_clearance")
    builder = GridQUBOBuilder(problem, penalties=PENALTIES)
    solver = DWaveSolver(normalize_scale=4, num_reads=100, seed=0)
    solution = solver.solve(builder, preprocess=True)

    assert math.isfinite(solution["energy"][-1]) if isinstance(solution["energy"], list) else True
    path = solver.decode_path(solution["solution"], problem)
    result = is_solution_valid(path, problem)
    assert result["valid"], result.get("message", result)


def test_preprocessing_routes_around_clearance_shell():
    """A parked robot sitting on the other robot's straight-line shortest
    path, one row off it, so every cell of that line is inside the pair's
    D_ab shell. The diagonal fixer used to pin that line anyway: its rival
    check was exact-cell only, and the parked robot is pinned by BFS, which
    the fixer never saw. With _rival_keepout (base_qubo.py) it detours around
    the shell, so nothing is forced and the result is valid."""
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=12,
                    robot_radius=0.6),
        RobotConfig("b", (1, 3), (1, 3), start_time=0, expected_duration=12,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="keepout_check")
    D = problem.get_clearance_table()[("a", "b")]
    reach = max(_cheb((0, 0), d) for d in D)

    for mode in ("full", "full_safe"):
        builder = GridQUBOBuilder(problem, penalties=PENALTIES)
        solver = DWaveSolver(normalize_scale=4, num_reads=50, seed=0)
        solution = solver.solve(builder, preprocess=mode)

        assert solution["metadata"]["forced_collisions"] == [], mode
        path = solver.decode_path(solution["solution"], problem)
        result = is_solution_valid(path, problem)
        assert result["valid"], (mode, result.get("message", result))
        a_cells = [cell[:2] for cell, robot in path if robot == 0]
        assert all(_cheb(c, (1, 3)) > reach for c in a_cells), (mode, a_cells)


def test_forced_clearance_violation_attributed_to_preprocessing():
    """_attribute_invalid_cause matches a clearance/trailing conflict against
    _flag_forced_collisions' log, so a violation preprocessing forced is
    blamed on pre_processing, not solver_sampling. Inputs are the ones the
    reproducer above produced before preprocessing became clearance-aware
    (robot numbers in the validation, robot ids in the log, as in a real run)."""
    validation = {
        "valid": False,
        "reason": "clearance_conflict+trailing_conflict",
        "details": {
            "clearance_conflicts": [
                {"cells": ((3, 1), (1, 3)), "time": 1, "robots": [0, 1]},
            ],
            "trailing_conflicts": [
                {"cells": ((3, 1), (1, 3)), "time": (1, 2), "robots": [0, 1]},
            ],
        },
    }
    forced_collisions = [
        {"kind": "clearance", "cells": ((1, 3), (3, 1)), "time": 1,
         "robots": ["a", "b"], "sources": [("a", "diag"), ("b", "bfs")],
         "origin": "same_window"},
        {"kind": "trailing", "cells": ((3, 1), (1, 3)), "time": (1, 2),
         "robots": ["a", "b"], "sources": [("a", "diag"), ("b", "bfs")],
         "origin": "same_window"},
    ]

    cause = _attribute_invalid_cause(validation, forced_collisions)
    assert cause["origin"] == "pre_processing", cause
    assert {m["kind"] for m in cause["matches"]} == {"clearance", "trailing"}

    unmatched = _attribute_invalid_cause(validation, [])
    assert unmatched == {"origin": "solver_sampling"}


def test_deadline_guard_stops_windowing_stall():
    """10x10 hard / four_robots: the windowed plan is locally valid every
    window but livelocks robot_1 away from its goal. _deadline_misses proves
    it can't make it and the solve stops early, attributed to windowing
    rather than solver_sampling."""
    problem = PathfindingProblem.from_map_config(
        "quantum/maps/synthetic/10x10/obs10x10_hard", "four_robots"
    ).as_grid_only()
    builder = GridQUBOBuilder(problem, penalties=PENALTIES)
    solver = DWaveSolver(normalize_scale=4, num_reads=4, seed=0)
    solution = solver.solve(builder, preprocess="full")

    misses = solution["metadata"]["deadline_misses"]
    assert misses and all(m["steps_needed"] > m["steps_left"] for m in misses)
    assert builder.current_T < builder.total_t  # stopped before the horizon

    result = is_solution_valid(solver.decode_path(solution["solution"], problem), problem)
    assert not result["valid"]
    cause = _attribute_invalid_cause(
        result,
        solution["metadata"]["forced_collisions"],
        window_stats=solution["metadata"]["window_stats"],
        deadline_misses=misses,
    )
    assert cause["origin"] == "windowing", cause


def test_deadline_guard_silent_on_solvable_problem():
    """No false alarms: a problem that solves validly never records a miss."""
    problem = PathfindingProblem.from_map_config(
        "quantum/maps/synthetic/10x10/obs10x10_easy", "four_robots"
    ).as_grid_only()
    builder = GridQUBOBuilder(problem, penalties=PENALTIES)
    solver = DWaveSolver(normalize_scale=4, num_reads=4, seed=0)
    solution = solver.solve(builder, preprocess="full")

    assert solution["metadata"]["deadline_misses"] == []
    result = is_solution_valid(solver.decode_path(solution["solution"], problem), problem)
    assert result["valid"], result.get("message")


def test_goal_not_reached_with_no_solver_windows_blamed_on_preprocessing():
    validation = {"valid": False, "reason": "robot_1_invalid", "details": {}}
    all_fixed = [{"final_variables": 0}, {"final_variables": 0}]
    cause = _attribute_invalid_cause(validation, [], window_stats=all_fixed)
    assert cause == {"origin": "pre_processing", "matches": [], "reason": "solver_never_ran"}

    some_solved = [{"final_variables": 0}, {"final_variables": 2}]
    cause = _attribute_invalid_cause(validation, [], window_stats=some_solved)
    assert cause == {"origin": "solver_sampling"}
