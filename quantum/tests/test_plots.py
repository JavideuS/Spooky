"""
quantum.benchmark.analysis.plots against the schema aggregate_sweep() emits.

These render real figures but never write files, so they stay fast. Their
value is catching schema drift: the plots consume runs_long, and a column
changing type there is invisible until someone runs the plotting CLI over a
finished sweep -- which is the worst moment to find out.
"""

import pandas as pd
import pytest

from quantum.benchmark.analysis import plots
from quantum.utils import preprocess as pm


def _runs_long(preprocess_values):
    """Minimal runs_long with the columns every plot reads."""
    rows = []
    for i, mode in enumerate(preprocess_values):
        for solver in ("sa_neal", "ilp_highs"):
            rows.append(
                {
                    "instance_map": "quantum/maps/synthetic/5x5/obs5x5_easy",
                    "problem_name": "two_robots",
                    "grid_size": "5x5",
                    "num_robots": 2,
                    "solver_name": solver,
                    "backend": "dwave" if solver == "sa_neal" else "ilp",
                    "preprocess": mode,
                    "valid": i % 2 == 0,
                    "energy": -3.0 - i,
                    "energy_excess": float(i),
                    "execution_time_sec": 0.1 * (i + 1),
                    "average_reduction_ratio": 0.5,
                    "total_initial_variables": 100,
                    "total_final_variables": 50,
                    "num_windows": 2,
                }
            )
    return pd.DataFrame(rows)


def test_plots_accept_preprocess_mode_strings():
    """Regression: `preprocess` became a mode string when the modes landed,
    and plot_variable_reduction still did `df["preprocess"] & ...`, which
    raised

      TypeError: unsupported operand type(s) for &: 'str' and 'bool'

    only when run over a real sweep -- every unit test until now built frames
    with booleans.
    """
    df = _runs_long([pm.FULL, pm.BFS_SAFE, pm.RAW, pm.BFS_AGGRESSIVE])
    fig = plots.plot_variable_reduction(df)
    assert fig is not None


def test_plots_still_accept_legacy_booleans():
    """Sweeps recorded before the modes existed have True/False in
    index.json, and must stay plottable."""
    fig = plots.plot_variable_reduction(_runs_long([True, False]))
    assert fig is not None


def test_raw_runs_are_excluded_from_variable_reduction():
    """`raw` does no pre-processing, so it has no reduction to chart -- and
    charting its rows would drag the mean toward zero for the wrong reason."""
    only_raw = _runs_long([pm.RAW])
    fig = plots.plot_variable_reduction(only_raw)
    assert all(len(trace.x) == 0 for trace in fig.data)


def test_generate_all_plots_covers_the_aggregate_schema():
    """The CLI path. If a plot reads a column aggregate_sweep() no longer
    emits, this is where it surfaces."""
    df = _runs_long([pm.FULL, pm.BFS_AGGRESSIVE])
    figs = plots.generate_all_plots(df, output_dir=None)
    assert {"scaling", "success_rate", "variable_reduction"} <= set(figs)


@pytest.mark.parametrize("mode", list(pm.MODES))
def test_every_mode_is_plottable(mode):
    fig = plots.plot_variable_reduction(_runs_long([mode]))
    assert fig is not None
