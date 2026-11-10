"""Committed paths (RobotConfig.fixed_path), CBS and ILP: the committed
steps are kept exactly, the robot is planned on from their end, everyone else
routes around them, and committing a robot to all of its own optimal plan
doesn't change the optimum. Colliding committed paths are rejected up front
by PathfindingProblem."""

import pytest

from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem, InfeasibleProblemError
from quantum.builder.CBSBuilder import GridCBSBuilder
from quantum.builder.ILPBuilder import GridILPBuilder
from quantum.solvers.CBS_solver import CBSSolver
from quantum.solvers.ILP_solver import ILPSolver
from quantum.benchmark.benchmark import is_solution_valid


SOLVERS = {
    "cbs": (lambda: CBSSolver(node_limit=5000, time_limit=20), GridCBSBuilder),
    "ilp": (lambda: ILPSolver(time_limit=20), GridILPBuilder),
}


@pytest.fixture(params=list(SOLVERS))
def kind(request):
    return request.param


def _solve(robots, grid, kind="cbs", **kw):
    problem = PathfindingProblem(robots, grid=grid, name="fixed_path", **kw)
    make_solver, builder_cls = SOLVERS[kind]
    solver = make_solver()
    solution = solver.solve(builder_cls(problem, name="fixed_path"), preprocess=True)
    return problem, solver, solution


def _cells(problem, rid):
    return [(i, j) for i, j, _ in sorted(problem.robots[rid].path, key=lambda p: p[2])]


def _assert_valid(problem, solver, solution):
    assert solution["metadata"]["termination_condition"] == "optimal"
    result = is_solution_valid(solver.decode_path(solution["solution"], problem), problem)
    assert result["valid"], result.get("message", result)


def test_prefix_is_kept_and_planned_on_from_its_end(kind):
    grid = Grid(3, 5, obstacles=[])
    # A detour nobody would pick: down, wait, back up, then on to the goal.
    prefix = [(0, 0), (1, 0), (1, 0), (0, 0)]
    a = RobotConfig("a", (0, 0), (0, 4), fixed_path=prefix)
    problem, solver, solution = _solve([a], grid, kind=kind)
    _assert_valid(problem, solver, solution)
    cells = _cells(problem, "a")
    assert cells[: len(prefix)] == prefix
    assert cells[len(prefix) - 1 : len(prefix) + 4] == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    # Cost counts from the prefix's first step: 3 committed + 4 moves.
    assert solution["energy"] == 7


def test_fully_fixed_robot_never_yields(kind):
    # 1-wide corridor with one side cell under (0, 2). 'b' is committed to
    # sweeping the corridor's middle: down into the side cell and back up
    # to park on (0, 2). Free, 'b' could let 'a' through; committed, it can't.
    grid = Grid(2, 5, obstacles=[(1, 0), (1, 1), (1, 3), (1, 4)])
    a = RobotConfig("a", (0, 0), (0, 4))
    b = RobotConfig("b", (1, 2), (0, 2), fixed_path=[(1, 2), (0, 2)])
    problem, solver, solution = _solve([a, b], grid, kind=kind)
    assert solution["metadata"]["termination_condition"] != "optimal"
    assert solution["energy"] == float("inf")

    b = RobotConfig("b", (1, 2), (0, 2))
    a = RobotConfig("a", (0, 0), (0, 4))
    _, _, solution = _solve([a, b], grid, kind=kind)
    assert solution["metadata"]["termination_condition"] == "optimal"


def test_others_route_around_a_committed_prefix(kind):
    # 'a' commits to sweeping row 1 left to right; 'b' crosses it top to
    # bottom in column 2 and must time its crossing around a's steps.
    grid = Grid(3, 5, obstacles=[])
    prefix = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    a = RobotConfig("a", (1, 0), (1, 4), fixed_path=prefix)
    b = RobotConfig("b", (0, 2), (2, 2))
    problem, solver, solution = _solve([a, b], grid, kind=kind)
    _assert_valid(problem, solver, solution)
    assert _cells(problem, "a")[: len(prefix)] == prefix
    pb = {t: (i, j) for i, j, t in problem.robots["b"].path}
    assert pb[2] != (1, 2)


def test_committing_an_optimal_plan_keeps_the_optimum(kind):
    grid = Grid(5, 5, obstacles=[(2, 2)])
    specs = [("a", (0, 0), (4, 4)), ("b", (4, 0), (0, 4)), ("c", (0, 4), (4, 0))]
    problem, solver, free = _solve([RobotConfig(*s) for s in specs], grid, kind=kind)
    _assert_valid(problem, solver, free)
    committed = _cells(problem, "a")
    first_goal = committed.index(specs[0][2])
    for cut in (3, first_goal + 1):  # partial prefix, then all of it
        robots = [RobotConfig(*specs[0], fixed_path=committed[:cut])]
        robots += [RobotConfig(*s) for s in specs[1:]]
        p2, s2, sol = _solve(robots, grid, kind=kind)
        _assert_valid(p2, s2, sol)
        assert sol["energy"] == free["energy"]
        assert _cells(p2, "a")[:cut] == committed[:cut]


def test_parked_robot_with_clearance_is_avoided(kind):
    # Same scene as test_cbs's clearance park, but 'b' is committed instead
    # of free: 'a' still keeps >= 3 cells (Chebyshev) away at every step.
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    a = RobotConfig("a", (3, 0), (3, 6), robot_radius=0.6)
    b = RobotConfig("b", (3, 3), (3, 3), robot_radius=0.6, fixed_path=[(3, 3)])
    problem, solver, solution = _solve([a, b], grid, kind=kind)
    _assert_valid(problem, solver, solution)
    pa = {t: (i, j) for i, j, t in problem.robots["a"].path}
    for t, (i, j) in pa.items():
        assert max(abs(i - 3), abs(j - 3)) >= 3, f"t={t}: ({i}, {j})"


@pytest.mark.parametrize(
    "a_path, b_path, clash",
    [
        # same cell at t=2
        ([(0, 0), (0, 1), (1, 1)], [(2, 1), (2, 1), (1, 1)], "a/b at t=2"),
        # swap across one edge at t=1 -> t=2
        ([(0, 0), (0, 1), (1, 1)], [(2, 1), (1, 1), (0, 1)], "a/b at t=1"),
    ],
)
def test_conflicting_committed_paths_are_rejected(a_path, b_path, clash):
    grid = Grid(3, 3, obstacles=[])
    a = RobotConfig("a", a_path[0], (2, 2), fixed_path=a_path)
    b = RobotConfig("b", b_path[0], (0, 0), fixed_path=b_path)
    with pytest.raises(InfeasibleProblemError, match=clash):
        PathfindingProblem([a, b], grid=grid)


def test_committed_path_through_a_robots_start_footprint_is_rejected():
    # Clearance pair (0.6 m on 0.4 m cells -> Chebyshev 2): 'a' commits to
    # driving past where 'b' appears at t=1, two cells away.
    grid = Grid(5, 5, obstacles=[], resolution=0.4)
    prefix = [(0, 0), (0, 1), (0, 2), (0, 3)]
    a = RobotConfig("a", (0, 0), (0, 4), robot_radius=0.6, fixed_path=prefix)
    b = RobotConfig("b", (2, 3), (4, 4), robot_radius=0.6, start_time=1)
    with pytest.raises(InfeasibleProblemError, match="a/b at t=1"):
        PathfindingProblem([a, b], grid=grid)


def test_fixed_path_must_start_at_start():
    grid = Grid(3, 3, obstacles=[])
    with pytest.raises(InfeasibleProblemError, match="fixed_path starts"):
        PathfindingProblem(
            [RobotConfig("a", (0, 0), (2, 2), fixed_path=[(0, 1), (0, 2)])], grid=grid
        )


def test_fixed_path_must_be_contiguous(kind):
    grid = Grid(3, 3, obstacles=[])
    with pytest.raises(InfeasibleProblemError, match="jumps"):
        _solve([RobotConfig("a", (0, 0), (2, 2), fixed_path=[(0, 0), (2, 0)])], grid, kind=kind)


def test_fixed_path_cell_on_obstacle_is_rejected():
    grid = Grid(3, 3, obstacles=[(1, 0)])
    with pytest.raises(InfeasibleProblemError, match="fixed_path"):
        PathfindingProblem(
            [RobotConfig("a", (0, 0), (2, 2), fixed_path=[(0, 0), (1, 0)])], grid=grid
        )


# --- QUBO (simulated annealing): heuristic, so validity and the committed
# steps are checked, not optimality. Committed steps reach the QUBO as
# forbidden cells for everyone else (BaseQUBO.committed_cells).

def _solve_qubo(robots, grid, preprocess=None, **kw):
    import yaml
    from pathlib import Path
    from quantum.builder.QUBOBuilder import GridQUBOBuilder
    from quantum.solvers.DWave_solver import DWaveSolver

    config = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    penalties = yaml.safe_load(config.read_text())["penalty_sets"]["crash"]
    problem = PathfindingProblem(robots, grid=grid, name="fixed_path_qubo", **kw)
    solver = DWaveSolver(seed=0, verbose_level=0)
    solution = solver.solve(GridQUBOBuilder(problem, penalties=penalties), preprocess=preprocess)
    result = is_solution_valid(solver.decode_path(solution["solution"], problem), problem)
    return problem, result


@pytest.mark.parametrize("preprocess", [None, "full"])
def test_qubo_keeps_prefix_and_routes_others_around_it(preprocess):
    grid = Grid(3, 5, obstacles=[])
    prefix = [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    a = RobotConfig("a", (1, 0), (1, 4), fixed_path=prefix)
    b = RobotConfig("b", (0, 2), (2, 2))
    problem, result = _solve_qubo([a, b], grid, preprocess)
    assert result["valid"], result.get("message", result)
    assert _cells(problem, "a")[: len(prefix)] == prefix


@pytest.mark.parametrize("preprocess", [None, "full"])
def test_qubo_partial_prefix_is_planned_on(preprocess):
    grid = Grid(3, 5, obstacles=[])
    prefix = [(0, 0), (1, 0), (1, 0), (0, 0)]
    a = RobotConfig("a", (0, 0), (0, 4), fixed_path=prefix)
    b = RobotConfig("b", (2, 4), (2, 0))
    problem, result = _solve_qubo([a, b], grid, preprocess)
    assert result["valid"], result.get("message", result)
    cells = _cells(problem, "a")
    assert cells[: len(prefix)] == prefix
    assert cells[-1] == (0, 4)


@pytest.mark.parametrize("preprocess", [None, "full"])
def test_qubo_parked_robot_with_clearance_is_avoided(preprocess):
    grid = Grid(7, 7, obstacles=[], resolution=0.4)
    a = RobotConfig("a", (3, 0), (3, 6), robot_radius=0.6)
    b = RobotConfig("b", (3, 3), (3, 3), robot_radius=0.6, fixed_path=[(3, 3)])
    problem, result = _solve_qubo([a, b], grid, preprocess)
    assert result["valid"], result.get("message", result)
    assert set(_cells(problem, "b")) == {(3, 3)}
    for i, j, t in problem.robots["a"].path:
        assert max(abs(i - 3), abs(j - 3)) >= 3, f"t={t}: ({i}, {j})"


def test_qubo_keeps_a_committed_cbs_plan():
    # clew's case: re-plan the others around a robot that keeps its plan.
    grid = Grid(5, 5, obstacles=[(2, 2)])
    specs = [("a", (0, 0), (4, 4)), ("b", (4, 0), (0, 4)), ("c", (0, 4), (4, 0))]
    problem, _, _ = _solve([RobotConfig(*s) for s in specs], grid)
    committed = _cells(problem, "a")
    robots = [RobotConfig(*specs[0], fixed_path=committed)]
    robots += [RobotConfig(*s) for s in specs[1:]]
    p2, result = _solve_qubo(robots, grid)
    assert result["valid"], result.get("message", result)
    assert _cells(p2, "a")[: len(committed)] == committed
