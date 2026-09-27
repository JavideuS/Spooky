"""
QUBOBuilder goal_distance="bfs": the goal-approach reward built on each
robot's obstacle-aware shortest-path distance instead of Manhattan.
"""

import pytest

from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem
from quantum.builder.QUBOBuilder import GridQUBOBuilder
from quantum.tests.test_qubo_clearance import PENALTIES


def _problem(obstacles):
    grid = Grid(5, 5, obstacles=obstacles)
    robots = [
        RobotConfig("a", (0, 0), (0, 4), start_time=0, expected_duration=14),
        RobotConfig("b", (4, 0), (4, 4), start_time=0, expected_duration=14),
    ]
    return PathfindingProblem(robots, grid=grid, name="goal_distance")


def test_bfs_equals_manhattan_without_obstacles():
    """On an open grid the shortest path is the Manhattan distance, so the
    two options must build byte-identical Q."""
    problem = _problem([])
    manhattan = GridQUBOBuilder(problem, penalties=PENALTIES)
    bfs = GridQUBOBuilder(problem, penalties=PENALTIES, goal_distance="bfs")
    manhattan.build()
    bfs.build()
    assert manhattan.Q == bfs.Q


def test_bfs_distance_goes_around_a_wall():
    """A wall down column 2 (open only at row 4): (0, 1) is 3 cells from the
    goal (0, 4) by Manhattan but 11 steps by the only route: down 4, under
    the wall 2, up 4, right 1."""
    wall = [(0, 2), (1, 2), (2, 2), (3, 2)]
    builder = GridQUBOBuilder(_problem(wall), penalties=PENALTIES, goal_distance="bfs")
    dist = builder.goal_distance_map("a")

    assert dist[(0, 4)] == 0
    assert dist[(0, 1)] == 11
    assert all(cell not in dist for cell in wall)
    # no local minima: every reachable non-goal cell has a neighbour one closer
    adjacency = builder.problem.grid.adjacency
    for cell, d in dist.items():
        if d:
            assert any(dist.get(n) == d - 1 for n in adjacency[cell]), cell


def test_unknown_goal_distance_rejected():
    with pytest.raises(ValueError):
        GridQUBOBuilder(_problem([]), penalties=PENALTIES, goal_distance="euclid")


def test_progress_weight_capped_below_collision_penalty():
    """The rule behind the cap: what a robot gives up by yielding one step
    must stay below the collision penalty it avoids, or the QUBO prefers
    colliding (or swapping) to yielding."""
    from quantum.builder.QUBOBuilder import PROGRESS_CAP_MARGIN

    builder = GridQUBOBuilder(_problem([]), penalties=PENALTIES, progress_weight=10.0)
    alpha = builder.effective_progress_weight()
    assert alpha == builder.progress_weight_cap() < 10.0
    # cost of yielding one step from the start: one cell behind at every later step
    yield_cost = sum(
        alpha * builder._goal_time_factor(t, 0) for t in range(1, builder.t_max)
    )
    assert yield_cost == pytest.approx(PROGRESS_CAP_MARGIN * PENALTIES["K_trail"])

    auto = GridQUBOBuilder(_problem([]), penalties=PENALTIES, progress_weight="auto")
    assert auto.effective_progress_weight() == pytest.approx(alpha)
    small = GridQUBOBuilder(_problem([]), penalties=PENALTIES, progress_weight=0.01)
    assert small.effective_progress_weight() == 0.01  # under the cap: untouched

    with pytest.raises(ValueError):
        GridQUBOBuilder(_problem([]), penalties=PENALTIES, progress_weight=-1)


def test_allow_wait_rewards_staying_and_drops_in_window_backtracking():
    """Waiting is enabled only for a robot with another robot near it (the
    only reason to wait is to yield); then staying is rewarded like a move and
    in-window backtracking pairs are dropped. A robot on its own keeps the
    strict terms."""
    grid = Grid(5, 5, obstacles=[])

    def build(b_start):
        robots = [
            RobotConfig("a", (0, 0), (0, 4), start_time=0, expected_duration=14),
            RobotConfig("b", b_start, (4, 4), start_time=0, expected_duration=14),
        ]
        problem = PathfindingProblem(robots, grid=grid, name="wait")
        builders = [
            GridQUBOBuilder(problem, penalties=PENALTIES, allow_wait=flag)
            for flag in (False, True)
        ]
        for b in builders:
            b.build()
        return builders

    N, MN = 5, 25
    stay = (0 * N + 1, MN + 0 * N + 1)  # robot "a" at (0, 1) for t=0 -> t=1
    move = (0 * N + 1, MN + 0 * N + 2)

    # b one cell away (radius is min(5, 5) // 2 = 2): a may wait
    off, on = build((1, 0))
    assert off.Q.get(stay, 0) == pytest.approx(PENALTIES["K_bt"])  # stay charged
    assert on.Q.get(stay, 0) == pytest.approx(-PENALTIES["K_adj"])  # stay = move
    assert on.Q[move] == off.Q[move] == pytest.approx(-PENALTIES["K_adj"])

    # b four cells away: nobody to yield to, strict terms unchanged
    off, on = build((4, 0))
    assert on.Q == off.Q
