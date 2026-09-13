"""
Phase 1 verification gate for the CBS classical baseline (see
/home/javideus/.claude/plans/giggly-dreaming-corbato.md):

1. CBS never returns a solution that violates is_solution_valid()'s two
   conflict checks (vertex, edge/swap) — including scenarios specifically
   built to force those conflicts if unhandled.
2. CBS and ILP agree on energy for shared tiny instances — the empirical
   confirmation that CBS's sum-of-costs objective and ILP's "timesteps away
   from goal" objective really are the same quantity, which is what the
   sweep pipeline's optimality-gap metric (Phase 3) depends on.
"""

import pytest

from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem, InfeasibleProblemError
from quantum.builder.CBSBuilder import GridCBSBuilder
from quantum.builder.ILPBuilder import GridILPBuilder
from quantum.solvers.CBS_solver import CBSSolver
from quantum.solvers.ILP_solver import ILPSolver
from quantum.solvers.cbs_algorithm import SpaceTimeAStar
from quantum.benchmark.benchmark import is_solution_valid
import networkx as nx


def _validate(solver, solution, problem):
    path = solver.decode_path(solution["solution"], problem)
    result = is_solution_valid(path, problem)
    assert result["valid"], result.get("message", result)


def test_cbs_corridor_forces_a_wait():
    """1x3 corridor: robot 'a' must cross it entirely (node0->node2); robot
    'b' only needs to duck one hop in (node2->node1) and is done by t=1 (its
    own short deadline — it stops existing/occupying anything after that,
    same active-window semantics as ILP). Planned independently (CBS's
    root), both would be at node1 at t=1 — a vertex conflict — so CBS must
    resolve it, and the only resolution here is 'a' waiting one step at
    node0 until 'b' has vacated (there's no room to detour in a 1-wide
    corridor, unlike test_cbs_vertex_conflict_resolved's 2x2 grid below).
    Note: a true end-to-end swap in a 1-wide corridor with no siding is
    mathematically impossible regardless of time slack (the two robots can
    never pass each other), which is why this fixture has 'b' duck in and
    finish early rather than also traverse the full corridor."""
    grid = Grid(1, 3, obstacles=[])
    robots = [
        RobotConfig("a", (0, 0), (0, 2), start_time=0, expected_duration=5),
        RobotConfig("b", (0, 2), (0, 1), start_time=0, expected_duration=2),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="corridor_wait")

    builder = GridCBSBuilder(problem, name="corridor_wait")
    solver = CBSSolver(node_limit=1000, time_limit=10)
    solution = solver.solve(builder, preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)
    # The resolution must actually be a wait, not a fluke: 'a' should still
    # be at its start cell at t=1 (one step later than its own root/
    # unconstrained plan would have it).
    assert problem.robots["a"].path[1][:2] == (0, 0)


def test_cbs_vertex_conflict_resolved():
    """2x2 grid, two robots whose shortest paths naturally collide at the
    same cell/time unless CBS branches on it."""
    grid = Grid(2, 2, obstacles=[])
    robots = [
        RobotConfig("a", (0, 0), (1, 1), start_time=0, expected_duration=6),
        RobotConfig("b", (0, 1), (1, 0), start_time=0, expected_duration=6),
    ]
    problem = PathfindingProblem(robots, grid=grid, T=6, name="vertex_conflict")

    builder = GridCBSBuilder(problem, name="vertex_conflict")
    solver = CBSSolver(node_limit=1000, time_limit=10)
    solution = solver.solve(builder, preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)


def test_cbs_single_robot_no_preprocess_still_valid():
    """preprocess=False should still produce a valid solution — it only
    widens the low-level search's candidate cells, never changes semantics."""
    grid = Grid(3, 3, obstacles=[(1, 1)])
    robot = RobotConfig("a", (0, 0), (2, 2), start_time=0, expected_duration=8)
    problem = PathfindingProblem(robot, grid=grid, T=8, name="single_no_preprocess")

    builder = GridCBSBuilder(problem, name="single_no_preprocess")
    solver = CBSSolver(node_limit=1000, time_limit=10)
    solution = solver.solve(builder, preprocess=False)

    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)


def _energies_match(map_path, problem_name):
    ilp_problem = PathfindingProblem.from_map_config(map_path, problem_name).as_grid_only()
    ilp_builder = GridILPBuilder(ilp_problem, name=problem_name)
    ilp_solver = ILPSolver(time_limit=30)
    ilp_solution = ilp_solver.solve(ilp_builder, preprocess=True)

    cbs_problem = PathfindingProblem.from_map_config(map_path, problem_name).as_grid_only()
    cbs_builder = GridCBSBuilder(cbs_problem, name=problem_name)
    cbs_solver = CBSSolver(node_limit=5000, time_limit=30)
    cbs_solution = cbs_solver.solve(cbs_builder, preprocess=True)

    assert ilp_solution["metadata"]["termination_condition"] == "optimal"
    assert cbs_solution["metadata"]["termination_condition"] == "optimal"
    _validate(ilp_solver, ilp_solution, ilp_problem)
    _validate(cbs_solver, cbs_solution, cbs_problem)
    assert abs(ilp_solution["energy"] - cbs_solution["energy"]) < 1e-6, (
        f"ILP energy {ilp_solution['energy']} != CBS energy {cbs_solution['energy']} "
        f"for {map_path}/{problem_name}"
    )


def test_cbs_ilp_energy_equivalence_single_robot():
    _energies_match("quantum/maps/synthetic/3x3/obs3x3_standard", "baseline")


def test_cbs_ilp_energy_equivalence_two_robots():
    _energies_match("quantum/maps/synthetic/5x5/obs5x5_easy", "two_robots")


def test_cbs_ilp_energy_equivalence_four_robots():
    _energies_match("quantum/maps/synthetic/10x10/obs10x10_hard", "four_robots")


def test_astar_refuses_arrival_it_cannot_hold():
    """A robot must not stop at goal if doing so would force it to occupy
    a forbidden (goal, t) cell later while padded/parked there — the
    padding-vs-constraint mismatch bug: find_path() used to return the
    naive first arrival regardless of later constraints, then the caller's
    unconditional padding step silently re-violated whatever constraint
    CBS had just added, producing a "conflict-free" child that still had
    the exact same conflict (an infinite/duplicate-child loop that burns
    node_limit without making progress). 1x3 corridor, goal is the middle
    node reachable at t=1 with plenty of deadline slack (5); a constraint
    at (goal, 3) must force a later arrival, not be silently ignored."""
    g = nx.Graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    astar = SpaceTimeAStar(g)

    path = astar.find_path(
        start=0, goal=1, start_time=0, deadline=5,
        forbidden_vertices={(1, 3)}, forbidden_edges=set(),
    )
    assert path is not None, "a later arrival avoiding (1,3) does exist here"

    arrival_time = path[-1][1]
    padded = list(path) + [(1, t) for t in range(arrival_time + 1, 6)]
    assert (1, 3) not in padded, (
        "padding re-introduced a cell the search was explicitly told to avoid"
    )


def test_cbs_robot_crossing_already_parked_goal_terminates_and_is_valid():
    """Full-solver version of the same scenario: robot 'a' must cross
    robot 'b's goal cell; 'b' parks there almost immediately, well before
    'a' would naturally arrive. Must terminate well under node_limit (no
    duplicate-child starvation) and produce a genuinely valid solution."""
    grid = Grid(1, 4, obstacles=[])
    robots = [
        RobotConfig("a", (0, 0), (0, 3), start_time=0, expected_duration=8),
        RobotConfig("b", (0, 1), (0, 2), start_time=0, expected_duration=3),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="cross_padded_goal")

    builder = GridCBSBuilder(problem, name="cross_padded_goal")
    solver = CBSSolver(node_limit=200, time_limit=5)
    solution = solver.solve(builder, preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    assert solution["raw_response"]["nodes_expanded"] < 50, (
        "search took far more nodes than this tiny instance should need — "
        "possible duplicate-child starvation regression"
    )
    _validate(solver, solution, problem)


def _cheb(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def test_cbs_clearance_enforces_combined_separation():
    """Robot 'a' crosses row 3 of a 7x7 grid at 0.4 m/cell while robot 'b'
    sits parked at (3, 3). Each clearance is 0.6 m -> footprint radius 1 cell
    -> D_ab spans Chebyshev 2, so 'a' must stay >= 3 cells from 'b' at every
    step (1.2 m), i.e. detour out to row 0 / row 6 around the parked robot
    rather than the one-row hop a single-cell CBS would take. Only 'a' has any
    freedom here, so there's no branching symmetry to blow up the search."""
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=24,
                    robot_radius=0.6),
        RobotConfig("b", (3, 3), (3, 3), start_time=0, expected_duration=24,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="clearance_park")

    builder = GridCBSBuilder(problem, name="clearance_park")
    solver = CBSSolver(node_limit=5000, time_limit=20)
    solution = solver.solve(builder, preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)

    pa = {t: (i, j) for i, j, t in problem.robots["a"].path}
    pb = {t: (i, j) for i, j, t in problem.robots["b"].path}
    assert set(pb.values()) == {(3, 3)}
    for t in set(pa) & set(pb):
        assert _cheb(pa[t], pb[t]) >= 3, f"t={t}: {pa[t]} vs {pb[t]} too close"


def test_cbs_default_radius_unchanged():
    """Same head-on, default robot_radius / unit resolution: footprint is a
    single cell, so the detour is the minimal one-row step and the leaders end
    up adjacent while passing — i.e. the clearance machinery is inert at the
    defaults."""
    grid = Grid(7, 7, obstacles=[])
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=24),
        RobotConfig("b", (3, 6), (3, 0), start_time=0, expected_duration=24),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="default_headon")
    assert problem.get_clearance_table()[("a", "b")] == frozenset({(0, 0)})

    builder = GridCBSBuilder(problem, name="default_headon")
    solver = CBSSolver(node_limit=5000, time_limit=20)
    solution = solver.solve(builder, preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)
    pa = {t: (i, j) for i, j, t in problem.robots["a"].path}
    pb = {t: (i, j) for i, j, t in problem.robots["b"].path}
    assert min(_cheb(pa[t], pb[t]) for t in set(pa) & set(pb)) == 1


def test_cbs_clearance_inflates_obstacles():
    """A robot with 0.6 m clearance on a 0.4 m grid keeps its whole path >= 2
    cells (Chebyshev) clear of an obstacle — except possibly its own start /
    goal, which inflation never removes."""
    grid = Grid(7, 7, obstacles=[(3, 3)], resolution=0.4)
    robot = RobotConfig("a", (0, 0), (6, 6), start_time=0, expected_duration=24,
                        robot_radius=0.6)
    problem = PathfindingProblem([robot], grid=grid, name="clearance_obstacle")

    builder = GridCBSBuilder(problem, name="clearance_obstacle")
    solver = CBSSolver(node_limit=5000, time_limit=20)
    solution = solver.solve(builder, preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)
    ends = {(0, 0), (6, 6)}
    for i, j, _t in problem.robots["a"].path:
        if (i, j) not in ends:
            assert _cheb((i, j), (3, 3)) >= 2, f"{(i, j)} hugs the obstacle"


def test_is_solution_valid_flags_clearance_violation():
    """A hand-fed plan where the two robots start and end comfortably apart
    (Chebyshev >= 3, clear of both the construction-time start- and
    goal-separation guards) but pass within their combined footprint
    (Chebyshev 2, inside 0.6 + 0.6 m at 0.4 m/cell) partway through — no cell
    ever shared, no swap. is_solution_valid must still reject it as a
    clearance conflict."""
    grid = Grid(7, 9, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (0, 0), (0, 4), start_time=0, expected_duration=8,
                    robot_radius=0.6),
        RobotConfig("b", (4, 4), (2, 7), start_time=0, expected_duration=8,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="clearance_validator")
    path = [
        ((0, 0, 0), 0), ((0, 1, 1), 0), ((0, 2, 2), 0), ((0, 3, 3), 0), ((0, 4, 4), 0),
        ((4, 4, 0), 1), ((3, 4, 1), 1), ((2, 4, 2), 1), ((2, 5, 3), 1),
        ((2, 6, 4), 1), ((2, 7, 5), 1),
    ]
    result = is_solution_valid(path, problem)
    assert not result["valid"]
    assert "clearance_conflict" in result["reason"]


def test_pair_separation_uses_max_inflation_not_sum():
    """Two robots, robot_radius 0.2 + inflation 0.3 each, 0.4 m grid. The
    required centre separation is 0.2 + 0.2 + max(0.3, 0.3) = 0.7 m -> cells
    within Chebyshev 1 are forbidden (>= 2 cells apart). The additive model
    would have used 0.8 + 0.8 = 1.6 m and forbidden Chebyshev 3."""
    from quantum.utils.clearance import pair_separation_offsets

    D = pair_separation_offsets(0.2, 0.3, 0.2, 0.3, resolution=0.4, factor=1.0)
    assert max(_cheb((0, 0), o) for o in D) == 1

    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=24,
                    robot_radius=0.2, inflation=0.3),
        RobotConfig("b", (3, 3), (3, 3), start_time=0, expected_duration=24,
                    robot_radius=0.2, inflation=0.3),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="max_inflation")
    solver = CBSSolver(node_limit=5000, time_limit=20)
    solution = solver.solve(GridCBSBuilder(problem, name="max_inflation"),
                            preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)
    pa = {t: (i, j) for i, j, t in problem.robots["a"].path}
    pb = {t: (i, j) for i, j, t in problem.robots["b"].path}
    assert min(_cheb(pa[t], pb[t]) for t in set(pa) & set(pb)) == 2


def test_goal_within_obstacle_clearance_is_rejected():
    grid = Grid(7, 7, obstacles=[(3, 3)], resolution=0.4)
    # goal (3, 4) is one cell from the obstacle; footprint radius is 1 at
    # robot_radius 0.6 / 0.4 m/cell, so the final approach can't be clear.
    robot = RobotConfig("a", (0, 0), (3, 4), start_time=0, expected_duration=20,
                        robot_radius=0.6)
    with pytest.raises(InfeasibleProblemError, match="final approach"):
        PathfindingProblem([robot], grid=grid, name="goal_in_inflation")


def test_start_within_obstacle_clearance_is_allowed():
    grid = Grid(7, 7, obstacles=[(3, 3)], resolution=0.4)
    robot = RobotConfig("a", (3, 4), (6, 6), start_time=0, expected_duration=20,
                        robot_radius=0.6)
    problem = PathfindingProblem([robot], grid=grid, name="start_in_inflation")
    solver = CBSSolver(node_limit=5000, time_limit=20)
    solution = solver.solve(GridCBSBuilder(problem, name="start_in_inflation"),
                            preprocess=True)
    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)


def test_goals_closer_than_separation_rejected_and_toggleable():
    grid = Grid(9, 9, obstacles=[], resolution=0.4)
    mk = lambda: [
        RobotConfig("a", (0, 0), (4, 4), start_time=0, expected_duration=20,
                    robot_radius=0.6),
        RobotConfig("b", (8, 8), (4, 6), start_time=0, expected_duration=20,
                    robot_radius=0.6),
    ]
    with pytest.raises(InfeasibleProblemError, match="cannot both park"):
        PathfindingProblem(mk(), grid=grid, name="goals_too_close")
    # clearance_enabled=False turns the whole clearance feature off
    PathfindingProblem(mk(), grid=grid, name="ok", clearance_enabled=False)


def test_starts_closer_than_separation_at_same_start_time_rejected():
    grid = Grid(9, 9, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (4, 4), (0, 0), start_time=0, expected_duration=20,
                    robot_radius=0.6),
        RobotConfig("b", (4, 6), (8, 8), start_time=0, expected_duration=20,
                    robot_radius=0.6),
    ]
    with pytest.raises(InfeasibleProblemError, match="cannot both be there at once"):
        PathfindingProblem(robots, grid=grid, name="starts_too_close")


def test_starts_closer_than_separation_but_different_start_time_allowed():
    """The provable case is only "same instant, both pinned" -- differing
    start_times are left to the solver (whether the earlier robot is still
    there depends on an arrival time not known before solving)."""
    grid = Grid(9, 9, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (4, 4), (0, 0), start_time=0, expected_duration=20,
                    robot_radius=0.6),
        RobotConfig("b", (4, 6), (8, 8), start_time=5, expected_duration=20,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="starts_close_staggered")
    assert problem.robots["b"].start_time == 5


def test_get_obstacle_keepout_shape():
    grid = Grid(7, 7, obstacles=[(3, 3)])  # unit resolution
    robots = [
        RobotConfig("a", (0, 0), (6, 6), robot_radius=1.5),   # footprint radius 1
        RobotConfig("b", (0, 6), (6, 0), robot_radius=0.5),   # footprint radius 0
    ]
    problem = PathfindingProblem(robots, grid=grid, name="keepout")
    keep = problem.get_obstacle_keepout()
    assert keep["b"] == frozenset()
    # the 8 cells ringing (3,3), minus the obstacle itself and a's start/goal
    assert keep["a"] == frozenset(
        (3 + di, 3 + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
    ) - {(3, 3)}
    problem.clearance_enabled = False
    problem._obstacle_keepout = None
    assert problem.get_obstacle_keepout()["a"] == frozenset()


def _clearance_park_problem():
    """Robot 'a' crosses row 3 of a 7x7, 0.4 m/cell grid while 'b' sits parked
    at (3, 3); robot_radius 0.6 -> D_ab spans Chebyshev 2, so 'a' must detour
    out to row 0 (>= 3 cells from (3,3)) to get past. Goals (3,0->3,6) and
    (3,3) are Chebyshev 3 apart, clear of the cross-goal guard."""
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=18,
                    robot_radius=0.6),
        RobotConfig("b", (3, 3), (3, 3), start_time=0, expected_duration=18,
                    robot_radius=0.6),
    ]
    return PathfindingProblem(robots, grid=grid, name="ilp_clearance_park")


def test_ilp_clearance_enforces_separation():
    problem = _clearance_park_problem()
    builder = GridILPBuilder(problem, name="ilp_clearance_park")
    solver = ILPSolver(time_limit=60)
    solution = solver.solve(builder, preprocess=True)

    assert solution["metadata"]["termination_condition"] == "optimal"
    # crash_clearance carries the separation here; trailing is 0 because
    # 'b' is pinned, so every would-be trailing violation is already a
    # same-time collision caught by crash / crash_clearance.
    assert len(builder.model.crash_clearance) > 0
    _validate(solver, solution, problem)

    pa = {t: (i, j) for i, j, t in problem.robots["a"].path}
    pb = {t: (i, j) for i, j, t in problem.robots["b"].path}
    assert set(pb.values()) == {(3, 3)}
    for t in set(pa) & set(pb):
        assert _cheb(pa[t], pb[t]) >= 3


def test_ilp_cbs_energy_equivalence_with_clearance():
    """The proof of concept: an independent exact solver reaches the same
    optimum as CBS on a clearance-constrained instance."""
    ilp_problem = _clearance_park_problem()
    ilp_sol = ILPSolver(time_limit=60).solve(
        GridILPBuilder(ilp_problem, name="ilp"), preprocess=True
    )
    cbs_problem = _clearance_park_problem()
    cbs_sol = CBSSolver(node_limit=5000, time_limit=30).solve(
        GridCBSBuilder(cbs_problem, name="cbs"), preprocess=True
    )
    assert ilp_sol["metadata"]["termination_condition"] == "optimal"
    assert cbs_sol["metadata"]["termination_condition"] == "optimal"
    assert abs(ilp_sol["energy"] - cbs_sol["energy"]) < 1e-6, (
        f"ILP {ilp_sol['energy']} != CBS {cbs_sol['energy']}"
    )


def test_ilp_trailing_populated_when_both_robots_move():
    """With both robots free to roam, trailing carries the genuine
    mid-transition crossings that no same-time check covers."""
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=18,
                    robot_radius=0.6),
        RobotConfig("b", (3, 3), (5, 0), start_time=0, expected_duration=18,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="ilp_both_move")
    builder = GridILPBuilder(problem, name="ilp_both_move")
    solution = ILPSolver(time_limit=90).solve(builder, preprocess=True)
    assert solution["metadata"]["termination_condition"] == "optimal"
    assert len(builder.model.trailing) > 0
    _validate(ILPSolver(), solution, problem)


def test_ilp_no_clearance_constraints_at_defaults():
    grid = Grid(5, 5, obstacles=[(2, 2)])
    robots = [
        RobotConfig("a", (0, 0), (4, 4), start_time=0, expected_duration=12),
        RobotConfig("b", (0, 4), (4, 0), start_time=0, expected_duration=12),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="ilp_default")
    builder = GridILPBuilder(problem, name="ilp_default")
    builder.build(preprocess=True)
    assert len(builder.model.crash_clearance) == 0
    assert len(builder.model.trailing) == 0
    assert builder.bfs_stats["obstacle_keepout_fixed"] == 0


def test_pair_separation_uses_max_not_sum_or_min_of_asymmetric_inflation():
    """a and b have deliberately different inflation (0.2 vs 0.6, radius 0.1
    each, 0.4 m/cell) -- three composition rules would disagree here:
      max(0.2,0.6) -> reach 0.8 -> forbidden radius 1 (separation >= 2)
      sum(0.2,0.6) -> reach 1.0 -> forbidden radius 2 (separation >= 3)
      min(0.2,0.6) -> reach 0.4 -> forbidden radius 0 (separation >= 1)
    so this pins down which rule is actually implemented, not just that
    *some* asymmetric-looking number comes out."""
    from quantum.utils.clearance import pair_separation_offsets

    D = pair_separation_offsets(0.1, 0.2, 0.1, 0.6, resolution=0.4, factor=1.0)
    assert max(_cheb((0, 0), o) for o in D) == 1
    # symmetric regardless of which robot is passed first
    D_swapped = pair_separation_offsets(0.1, 0.6, 0.1, 0.2, resolution=0.4, factor=1.0)
    assert D == D_swapped

    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 6), start_time=0, expected_duration=18,
                    robot_radius=0.1, inflation=0.2),
        RobotConfig("b", (3, 3), (3, 3), start_time=0, expected_duration=18,
                    robot_radius=0.1, inflation=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="asymmetric_inflation")
    solver = CBSSolver(node_limit=5000, time_limit=20)
    solution = solver.solve(GridCBSBuilder(problem, name="asymmetric_inflation"),
                            preprocess=True)
    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)
    pa = {t: (i, j) for i, j, t in problem.robots["a"].path}
    pb = {t: (i, j) for i, j, t in problem.robots["b"].path}
    assert min(_cheb(pa[t], pb[t]) for t in set(pa) & set(pb)) >= 2


def test_separation_factor_widens_required_gap():
    """factor > 1 (odom/lidar slack) must strictly widen the forbidden
    region versus factor == 1, for the exact same robots."""
    from quantum.utils.clearance import pair_separation_offsets

    D_default = pair_separation_offsets(0.3, 0.3, 0.3, 0.3, resolution=0.4, factor=1.0)
    D_padded = pair_separation_offsets(0.3, 0.3, 0.3, 0.3, resolution=0.4, factor=3.0)
    assert D_padded.issuperset(D_default)
    assert D_padded != D_default

    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (0, 0), (1, 1), robot_radius=0.3, inflation=0.3),
        RobotConfig("b", (6, 6), (5, 5), robot_radius=0.3, inflation=0.3),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="factor_wired",
                                 separation_factor=3.0)
    manual = pair_separation_offsets(0.3, 0.3, 0.3, 0.3, resolution=0.4, factor=3.0)
    assert problem.get_clearance_table()[("a", "b")] == manual


def test_three_mutually_clearance_robots_all_pairs_separated():
    """Three robots, all pairwise clearance pairs: get_clearance_table() must carry all
    3*2=6 ordered pairs, and CBS must respect every pair's separation
    simultaneously, not just the pair it happened to branch on first."""
    # b, c parked far enough from a's start/goal and from each other (all
    # pairwise Chebyshev >= 4, comfortably clear of the required >= 3) that
    # only a's mid-transit approach to each of them needs a detour.
    grid = Grid(7, 13, obstacles=[], resolution=0.4)
    robots = [
        RobotConfig("a", (3, 0), (3, 12), start_time=0, expected_duration=30,
                    robot_radius=0.6),
        RobotConfig("b", (3, 4), (3, 4), start_time=0, expected_duration=30,
                    robot_radius=0.6),
        RobotConfig("c", (3, 8), (3, 8), start_time=0, expected_duration=30,
                    robot_radius=0.6),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="three_clearance")
    table = problem.get_clearance_table()
    assert len(table) == 6
    assert all(off != frozenset({(0, 0)}) for off in table.values())

    solver = CBSSolver(node_limit=5000, time_limit=20)
    solution = solver.solve(GridCBSBuilder(problem, name="three_clearance"), preprocess=True)
    assert solution["metadata"]["termination_condition"] == "optimal"
    _validate(solver, solution, problem)

    paths = {rid: {t: (i, j) for i, j, t in r.path} for rid, r in problem.robots.items()}
    for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
        shared = set(paths[x]) & set(paths[y])
        assert min(_cheb(paths[x][t], paths[y][t]) for t in shared) >= 3
