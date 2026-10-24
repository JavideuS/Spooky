"""
The grid builder's arrival-time goal term: the BFS goal distance it is built
on, and the encodings around it that the diagonal fixer depends on.
"""

import pytest

from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem
from quantum.builder.QUBOBuilder import GridQUBOBuilder
from quantum.tests.test_qubo_clearance import PENALTIES


def _problem(obstacles, goal_a=(0, 4)):
    grid = Grid(5, 5, obstacles=obstacles)
    robots = [
        RobotConfig("a", (0, 0), goal_a, start_time=0, expected_duration=14),
        RobotConfig("b", (4, 0), (4, 4), start_time=0, expected_duration=14),
    ]
    return PathfindingProblem(robots, grid=grid, name="goal_distance")


def test_bfs_distance_goes_around_a_wall():
    """A wall down column 2 (open only at row 4): (0, 1) is 3 cells from the
    goal (0, 4) by Manhattan but 11 steps by the only route: down 4, under
    the wall 2, up 4, right 1."""
    wall = [(0, 2), (1, 2), (2, 2), (3, 2)]
    builder = GridQUBOBuilder(_problem(wall), penalties=PENALTIES)
    dist = builder.goal_distance_map("a")

    assert dist[(0, 4)] == 0
    assert dist[(0, 1)] == 11
    assert all(cell not in dist for cell in wall)
    # no local minima: every reachable non-goal cell has a neighbour one closer
    adjacency = builder.problem.grid.adjacency
    for cell, d in dist.items():
        if d:
            assert any(dist.get(n) == d - 1 for n in adjacency[cell]), cell


def test_invalid_goal_options_rejected():
    with pytest.raises(ValueError):
        GridQUBOBuilder(_problem([]), penalties=PENALTIES, goal_objective="legacy")
    with pytest.raises(ValueError):
        GridQUBOBuilder(_problem([]), penalties=PENALTIES, goal_weight=0)


def test_staying_rewarded_only_on_the_goal():
    """grid.adjacency has no self-move, so a stay would pay K_adj; on the goal
    it is rewarded like a move (else the robot steps off its goal), anywhere
    else it keeps that cost."""
    N, MN = 5, 25
    b = GridQUBOBuilder(_problem([], goal_a=(0, 1)), penalties=PENALTIES)
    b.build(["adjacency_reward"])
    stay_on_goal = (0 * N + 1, MN + 0 * N + 1)  # a at its goal (0, 1), t=0 -> 1
    stay_elsewhere = (0 * N + 0, MN + 0 * N + 0)  # a at (0, 0), t=0 -> 1
    assert b.Q[stay_on_goal] == pytest.approx(-PENALTIES["K_adj"])
    assert stay_elsewhere not in b.Q


def test_lock_has_no_diagonal():
    """The lock is written as goal-then-any-other-cell pairs: the old
    +K_lock on the goal's diagonal made the diagonal fixer read the goal as
    the most expensive cell and pin it to 0."""
    b = GridQUBOBuilder(_problem([]), penalties=PENALTIES)
    b.build(["lock"])
    assert b.Q and all(i != j for i, j in b.Q)
    assert all(v == pytest.approx(PENALTIES["K_lock"]) for v in b.Q.values())


def test_contested_fixer_leaves_the_yield_to_the_sampler():
    """obs5x5_hard two_robots: both robots' greedy paths meet head-on in
    row 0. The chain fixer pins robot_0's straight path first, so robot_1
    flees into the dead end (4, 3) that robot_0 then parks in front of;
    fixer="contested" leaves those steps to the sampler."""
    from quantum.solvers.DWave_solver import DWaveSolver
    from quantum.benchmark.benchmark import is_solution_valid

    for fixer, expect in (("chain", False), ("contested", True)):
        problem = PathfindingProblem.from_map_config(
            "quantum/maps/synthetic/5x5/obs5x5_hard", "two_robots"
        ).as_grid_only()
        builder = GridQUBOBuilder(problem, penalties=PENALTIES, fixer=fixer)
        solver = DWaveSolver(normalize_scale=1, num_reads="auto", seed=1)
        solution = solver.solve(builder, preprocess="full")
        path = solver.decode_path(solution["solution"], problem)
        assert is_solution_valid(path, problem)["valid"] is expect, fixer
