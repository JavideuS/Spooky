"""
Coverage for PathfindingProblem's construction-time feasibility guards: a
robot's start/goal that is out of bounds, on an obstacle, or an unknown
graph node, or a start_time that leaves no time to plan within an explicit
horizon, must be rejected at construction time (InfeasibleProblemError),
not discovered later as a cryptic var_limit crash, a silently-unsatisfiable
QUBO, or a failed benchmark run.
"""

import pytest

from quantum.map import Grid, Graph
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem, InfeasibleProblemError


def _grid():
    return Grid(3, 3, obstacles=[(1, 1)])


def test_valid_start_and_goal_construct_without_error():
    robot = RobotConfig("a", start=(0, 0), goal=(2, 2))
    problem = PathfindingProblem(robot, grid=_grid())
    assert problem.robots["a"].start == (0, 0)


def test_start_on_obstacle_raises():
    robot = RobotConfig("a", start=(1, 1), goal=(2, 2))
    with pytest.raises(InfeasibleProblemError, match="obstacle"):
        PathfindingProblem(robot, grid=_grid())


def test_goal_on_obstacle_raises():
    robot = RobotConfig("a", start=(0, 0), goal=(1, 1))
    with pytest.raises(InfeasibleProblemError, match="obstacle"):
        PathfindingProblem(robot, grid=_grid())


@pytest.mark.parametrize("start", [(-1, 0), (3, 0), (0, -1), (0, 3)])
def test_out_of_bounds_start_raises(start):
    robot = RobotConfig("a", start=start, goal=(2, 2))
    with pytest.raises(InfeasibleProblemError, match="out of bounds"):
        PathfindingProblem(robot, grid=_grid())


def test_out_of_bounds_goal_raises():
    robot = RobotConfig("a", start=(0, 0), goal=(5, 5))
    with pytest.raises(InfeasibleProblemError, match="out of bounds"):
        PathfindingProblem(robot, grid=_grid())


def test_one_bad_robot_among_several_still_raises():
    robots = [
        RobotConfig("good", start=(0, 0), goal=(0, 2)),
        RobotConfig("bad", start=(1, 1), goal=(2, 0)),
    ]
    with pytest.raises(InfeasibleProblemError, match="'bad'"):
        PathfindingProblem(robots, grid=_grid())


def _graph():
    return Graph(nodes=[(0, 0), (0, 1), (1, 1)], edges=[(0, 1), (1, 2)])


def test_valid_graph_node_ids_construct_without_error():
    robot = RobotConfig("a", start=0, goal=2)
    problem = PathfindingProblem(robot, graph=_graph())
    assert problem.robots["a"].goal == 2


@pytest.mark.parametrize("goal", [-1, 3, 999])
def test_out_of_range_graph_node_id_raises(goal):
    robot = RobotConfig("a", start=0, goal=goal)
    with pytest.raises(InfeasibleProblemError, match="does not exist"):
        PathfindingProblem(robot, graph=_graph())


def test_start_time_beyond_horizon_raises():
    robot = RobotConfig("a", start=(0, 0), goal=(2, 2), start_time=5)
    with pytest.raises(InfeasibleProblemError, match="leaves no time"):
        PathfindingProblem(robot, grid=_grid(), T=5)


def test_start_time_at_horizon_raises():
    """start_time == T leaves a zero-length window -- also infeasible, not
    just start_time > T."""
    robot = RobotConfig("a", start=(0, 0), goal=(2, 2), start_time=4)
    with pytest.raises(InfeasibleProblemError, match="leaves no time"):
        PathfindingProblem(robot, grid=_grid(), T=4)


def test_start_time_within_horizon_is_fine():
    robot = RobotConfig("a", start=(0, 0), goal=(2, 2), start_time=2)
    problem = PathfindingProblem(robot, grid=_grid(), T=6)
    assert problem.robots["a"].T == 4
