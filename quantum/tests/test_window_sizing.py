"""
BaseQUBO.fit_window_to_var_limit() sizes each window
from the real BFS layers instead of max_window_size()'s connectivity**step
estimate. Builder-level only -- no solver runs, so these stay fast.
"""

import pytest

from quantum.builder.QUBOBuilder import GridQUBOBuilder
from quantum.map import Grid
from quantum.pathFormulation import PathfindingProblem
from quantum.robotConfiguration import RobotConfig
from quantum.utils import preprocess as pm

PENALTIES = {"K_hot": 1, "K_adj": 1, "K_start": 1, "K_goal": 1}


def _builder(var_limit, **kwargs):
    grid = Grid(8, 8, obstacles=[(3, 3), (3, 4), (4, 3)])
    robots = [
        RobotConfig("a", (0, 0), (7, 7), start_time=0, expected_duration=20),
        RobotConfig("b", (7, 0), (0, 7), start_time=0, expected_duration=20),
    ]
    problem = PathfindingProblem(robots, grid=grid, name="sizing")
    return GridQUBOBuilder(
        problem,
        penalties=PENALTIES,
        var_limit=var_limit,
        **{"window_sizing": "bfs", **kwargs},
    )


def _count(active_cells):
    return sum(len(cells) for cells in active_cells.values())


@pytest.mark.parametrize(
    "variant", [pm.bfs_variant(pm.FULL), pm.bfs_variant(pm.FULL_SAFE)]
)
@pytest.mark.parametrize("var_limit", [20, 60, 150])
def test_window_is_the_largest_that_fits(variant, var_limit):
    builder = _builder(var_limit)
    _, cells = builder.fit_window_to_var_limit(variant)
    window = builder.t_max
    assert _count(cells) <= var_limit
    # Robots have no cells past the horizon, so the last window (total_t + 1,
    # see max_window_size()) covers one step fewer than its size.
    assert {t for _, t in cells} == set(range(min(window, builder.total_t)))

    # One more step must either overflow or be past the ceiling.
    builder.t_max = window + 1
    _, longer = builder.get_logical_variables(variant)
    assert _count(longer) > var_limit or window + 1 > builder.total_t + 1


@pytest.mark.parametrize(
    "variant", [pm.bfs_variant(pm.FULL), pm.bfs_variant(pm.FULL_SAFE)]
)
def test_truncation_matches_a_direct_build(variant):
    """The single BFS over the longest window, truncated, must equal what
    get_logical_variables() gives for the chosen window directly -- that is
    what makes skipping the second BFS safe."""
    builder = _builder(60)
    fixed, cells = builder.fit_window_to_var_limit(variant)
    direct_fixed, direct_cells = builder.get_logical_variables(variant)
    assert fixed == direct_fixed
    assert cells == direct_cells


def test_caps_still_bind():
    builder = _builder(10_000, window_max_steps=4)
    builder.fit_window_to_var_limit()
    assert builder.t_max == 4


def test_window_sizing_is_validated():
    with pytest.raises(ValueError):
        _builder(60, window_sizing="exact")
