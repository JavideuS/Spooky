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

    NOT covered by this test (found while writing it, not something these
    changes introduced -- see base_qubo.py's _collides_with_swap_rival
    docstring): when BFS/diagonal preprocessing manages to fully determine a
    window on its own ("fully pre-processed, skipping solver"), it can
    commit two robots to positions that violate D_ab-shell clearance without
    ever consulting K_crash/K_trail, because that preprocessing pass is
    exact-cell/exact-swap only. Reproduced with a parked robot sitting
    exactly on the other robot's straight-line shortest path -- confirmed
    present at default (no-clearance) radius too, so it's a pre-existing
    windowing gap, not a clearance regression. Not fixed here; flagged as a
    known gap in clearance.md / base_qubo.py."""
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


def test_forced_clearance_violation_attributed_to_preprocessing():
    """The gap from the previous test, now diagnosed rather than silent:
    _flag_forced_collisions (base_solver.py) is clearance-shell aware, so a
    window preprocessing fully resolves on its own can still be caught and
    correctly attributed to pre_processing rather than solver_sampling, even
    though preprocessing itself still isn't clearance-aware (that part is
    still open -- this only fixes the after-the-fact accounting). Same
    reproducer as above (parked robot sitting on the other's straight-line
    shortest path) but at 'b' offset one row up, which used to leave
    forced_collisions empty."""
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=12,
                    robot_radius=0.6),
        RobotConfig("b", (1, 3), (1, 3), start_time=0, expected_duration=12,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="attribution_check")
    builder = GridQUBOBuilder(problem, penalties=PENALTIES)
    solver = DWaveSolver(normalize_scale=4, num_reads=50, seed=0)
    solution = solver.solve(builder, preprocess=True)

    path = solver.decode_path(solution["solution"], problem)
    result = is_solution_valid(path, problem)
    assert not result["valid"]  # still the same underlying gap, unfixed
    assert "clearance_conflict" in result["reason"] or "trailing_conflict" in result["reason"]

    forced_collisions = solution["metadata"]["forced_collisions"]
    assert any(fc["kind"] in ("clearance", "trailing") for fc in forced_collisions), (
        "expected at least one clearance/trailing entry in forced_collisions"
    )

    cause = _attribute_invalid_cause(result, forced_collisions)
    assert cause["origin"] == "pre_processing", cause
    assert any(m["kind"] in ("clearance", "trailing") for m in cause["matches"])
