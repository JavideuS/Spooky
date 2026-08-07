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

from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem
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
