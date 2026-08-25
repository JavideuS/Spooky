"""
Coverage for the Manhattan-distance-vs-T horizon check, which is treated
differently depending on the solver family:

- ILP/CBS (validate_time_horizon() in quantum/builder/ILPBuilder.py) solve
  the whole horizon in one shot, with no windowing to fall back on, so a
  robot's T that's too small to possibly reach its goal (fewer than
  Manhattan-distance moves available) is a hard InfeasibleProblemError at
  builder construction.
- QUBO/QAOA windowed builders (BaseQUBO._warn_if_horizon_too_tight() in
  quantum/builder/base_qubo.py) only warn via the logger: a tight T there
  can be a deliberate approximation-under-budget experiment rather than a
  mistake, so construction must still succeed.
"""

import pytest

from quantum.map import Grid
from quantum.robotConfiguration import RobotConfig
from quantum.pathFormulation import PathfindingProblem, InfeasibleProblemError
from quantum.builder.ILPBuilder import GridILPBuilder
from quantum.builder.CBSBuilder import GridCBSBuilder
from quantum.builder.QUBOBuilder import GridQUBOBuilder
from quantum.utils.logger import set_verbose_level, get_logger, VerboseLogger


def _grid():
    return Grid(5, 5, obstacles=[])


def _too_short_problem():
    # Manhattan distance (0,0)->(4,4) is 8; T=3 gives only 2 moves.
    robot = RobotConfig("a", start=(0, 0), goal=(4, 4), expected_duration=3)
    return PathfindingProblem(robot, grid=_grid())


def _exactly_enough_problem():
    # T-1 moves must be >= dist; T=9 gives exactly 8 moves for dist=8.
    robot = RobotConfig("a", start=(0, 0), goal=(4, 4), expected_duration=9)
    return PathfindingProblem(robot, grid=_grid())


@pytest.mark.parametrize("builder_cls", [GridILPBuilder, GridCBSBuilder])
def test_exact_solvers_reject_too_short_horizon(builder_cls):
    with pytest.raises(InfeasibleProblemError, match="Manhattan distance"):
        builder_cls(_too_short_problem())


@pytest.mark.parametrize("builder_cls", [GridILPBuilder, GridCBSBuilder])
def test_exact_solvers_accept_exactly_enough_horizon(builder_cls):
    builder_cls(_exactly_enough_problem())  # must not raise


def test_qubo_builder_does_not_hard_fail_on_too_short_horizon():
    """Windowed QUBO builders get no hard check here -- a tight T can be a
    deliberate approximation-under-budget experiment, not just a mistake."""
    GridQUBOBuilder(_too_short_problem(), penalties={"name": "test"})


@pytest.fixture
def restore_verbose_level():
    original = get_logger().level
    yield
    set_verbose_level(original)


def test_qubo_builder_warns_on_too_short_horizon(capsys, restore_verbose_level):
    set_verbose_level(VerboseLogger.MINIMAL)
    GridQUBOBuilder(_too_short_problem(), penalties={"name": "test"})
    out = capsys.readouterr().out
    assert "[WARNING]" in out
    assert "Robot 'a'" in out
    assert "Manhattan distance" in out


def test_qubo_builder_does_not_warn_with_sufficient_horizon(capsys, restore_verbose_level):
    set_verbose_level(VerboseLogger.MINIMAL)
    GridQUBOBuilder(_exactly_enough_problem(), penalties={"name": "test"})
    out = capsys.readouterr().out
    assert "[WARNING]" not in out
