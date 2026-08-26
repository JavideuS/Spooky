"""
How large the QUBO actually is, per pre-processing mode, without solving it.

A sweep can only report variable counts for runs that completed, which biases
the numbers exactly where they are most interesting: `raw` is the pruning
baseline, and it is also the mode most likely to exhaust a simulator before
recording anything. Every reduction ratio in the sweep output is relative to
a baseline that therefore goes unmeasured.

Three reductions are easy to conflate, so the columns keep them apart:

  encoded_variables   the naive flat encoding, robots x cells x horizon.
                      Identical for every mode -- pruning removes variables
                      from a window, never from the problem.
  window_variables    what one window actually holds. `raw` is NOT the
                      unpruned problem: windowing alone already cut it
                      (5x5/two_robots is 750 encoded but a 250-variable raw
                      window), so the raw row's fraction_of_encoding is the
                      windowing reduction on its own, before any BFS runs.
  vs_raw              pruning alone, since numerator and denominator are both
                      windowed. This is the honest BFS/numerical attribution.

window_variables is a *peak* -- the solver holds one window at a time, which
is what a qubit budget is spent on. total_window_variables sums across the
solve, which is what compute is spent on, and full_horizon_variables is what
a single un-windowed QUBO would cost. windowing_gain is the ratio of the last
two: above 1 means windowing pays for itself in total variables, not only in
peak.

Whether it does depends entirely on the pruning, and the answer is not the
obvious one:

  raw          windowing costs MORE in total (5x5/two_robots: 1000 across
               four windows against 750 un-windowed). Consecutive windows
               overlap by a timestep and re-cover the same cells, and with no
               pruning there is nothing to offset that.
  bfs_safe     windowing wins decisively -- 264 against 506 on the same
               instance, 600 against 4984 on 10x10/four_robots. A monotone
               reachable set saturates to the whole free space over a long
               horizon and stays there; windowing re-seeds it from
               robot.current_position every couple of steps
               (QUBOBuilder.get_logical_variables), so it never leaves the
               early-steps regime where BFS actually prunes.

The aggressive variant is not a useful guide here: un-windowed it reports 58
against windowed 112, but only because its global `visited` set lets a cell
appear once across the entire horizon. That is an unsound bound rather than a
reduction, and comparing against it makes windowing look wasteful when it is
not.

Windowing therefore helps both the peak and the total -- but each window is
solved without sight of what follows, so it trades solution quality for both.
A machine that can take the full horizon in one shot should, for that reason
and not for variable count.

Building a QUBO invokes no solver, so it cannot run out of qubits and it
takes milliseconds. That makes the scaling curve measurable across instances
no backend can reach -- which is the point: knowing that 10x10/four_robots is
304 variables reduced to 8 says more about what pre-processing is doing than
any success rate does.

The measurement mirrors BaseSolver._prepare_window()'s sequence exactly --
get_logical_variables(bfs_variant) -> build() -> optional
reduce_diag_fixed_vars_iterative() -- so the counts are the ones the solver
would have seen, not an independent reimplementation that could drift.

Every function here is plain and independently importable; see
run_variable_scaling.py for the CLI wrapper.
"""

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from quantum.builder.GraphQUBO import GraphQUBO
from quantum.builder.QUBOBuilder import QUBOBuilder
from quantum.pathFormulation import PathfindingProblem
from quantum.utils import preprocess as preprocess_modes

# Modes worth charting by default. full_safe is omitted because it collapses
# to the same window as full on every instance measured so far; pass it
# explicitly if you want to confirm that on a new map.
# var_limit large enough that max_window_size() spans any horizon, giving the
# single un-windowed QUBO that windowing is measured against
_UNWINDOWED_VAR_LIMIT = 10**9

DEFAULT_MODES: Tuple[str, ...] = (
    preprocess_modes.RAW,
    preprocess_modes.BFS_AGGRESSIVE,
    preprocess_modes.BFS_SAFE,
    preprocess_modes.FULL,
)


def _build_for_mode(problem, penalties, mode, var_limit=None):
    """One builder taken through the same stages _prepare_window() would.

    Returns (builder, window_vars_before_numeric, window_vars_after_numeric).
    Under `raw` no BFS runs, so _active_cells stays None and build() produces
    the unpruned *window* -- still windowed, just not pruned. That is the
    baseline vs_raw is measured against.
    """
    kwargs: Dict[str, Any] = {"verbose_level": 0}
    if var_limit is not None:
        kwargs["var_limit"] = var_limit

    is_grid = problem.get_format_type() != "graph"
    scoped = problem.as_grid_only() if is_grid else problem.as_graph_only()
    builder_cls = QUBOBuilder if is_grid else GraphQUBO
    builder = builder_cls(scoped, penalties, **kwargs)

    variant = preprocess_modes.bfs_variant(mode)
    if variant is not None:
        _, active_cells = builder.get_logical_variables(variant)
        builder._active_cells = active_cells

    builder.build()
    before = builder.get_num_wires()

    if preprocess_modes.applies_numeric_reduction(mode):
        builder.reduce_diag_fixed_vars_iterative()
    after = builder.get_num_wires()

    return builder, before, after


def measure_instance(
    map_path: str,
    problem_name: str,
    penalties: Dict[str, Any],
    modes: Sequence[str] = DEFAULT_MODES,
    var_limit: Optional[int] = None,
    measure_full_horizon: bool = True,
) -> List[Dict[str, Any]]:
    """One row per mode for a single (map, problem) pair.

    encoded_variables is the full flat encoding -- robots x cells x horizon --
    and is the same for every mode, since pruning removes variables from the
    window rather than from the problem. It is the denominator that makes
    window_variables comparable across instances of different sizes.
    """
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        mode = preprocess_modes.normalize(mode)
        try:
            builder, before, after = _build_for_mode(
                PathfindingProblem.from_map_config(map_path, problem_name),
                dict(penalties),
                mode,
                var_limit,
            )
        except Exception as exc:  # a map/problem that cannot even be built
            rows.append(
                {
                    "instance_map": map_path,
                    "problem_name": problem_name,
                    "preprocess": mode,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        encoded = builder.initial_num_vars
        window_steps = builder.max_window_size()

        full_horizon = None
        if measure_full_horizon:
            try:
                # var_limit high enough that max_window_size() spans the whole
                # horizon, i.e. one un-windowed QUBO. Cheap even at 10x10
                # (~190k Q terms, 0.09s), but it is the one measurement here
                # that scales with the map, so it can be turned off.
                _, fh_before, fh_after = _build_for_mode(
                    PathfindingProblem.from_map_config(map_path, problem_name),
                    dict(penalties),
                    mode,
                    _UNWINDOWED_VAR_LIMIT,
                )
                full_horizon = fh_after
            except Exception:  # too large to build; the ratio is just absent
                full_horizon = None

        # BaseQUBO.update_problem() advances current_T by t_max - 1, so
        # consecutive windows share their boundary timestep
        stride = max(window_steps - 1, 1)
        num_windows = max(1, math.ceil(max(builder.total_t - 1, 1) / stride))
        rows.append(
            {
                "instance_map": map_path,
                "problem_name": problem_name,
                "preprocess": mode,
                "num_robots": builder.problem.num_robots,
                "horizon": builder.total_t,
                "window_max_steps": window_steps,
                "num_windows": num_windows,
                "encoded_variables": encoded,
                "window_variables": before,
                "window_variables_after_numeric": after,
                # Extrapolated from window 0, not measured per window: it
                # assumes every window looks like the first. Each window
                # re-seeds its BFS from a single cell, so that holds well for
                # the BFS stage; the numerical stage can vary more, and a
                # `full` row reading 0 means window 0 reduced away entirely,
                # not that every window does. Peak vs total: see the module
                # docstring.
                "total_window_variables": after * num_windows,
                "full_horizon_variables": full_horizon,
                # > 1 means windowing costs fewer total variables than one
                # un-windowed QUBO, not merely a smaller peak
                "windowing_gain": (
                    full_horizon / (after * num_windows)
                    if full_horizon and after * num_windows
                    else None
                ),
                "fraction_of_encoding": (before / encoded) if encoded else None,
                "error": None,
            }
        )
    return rows


def measure_scaling(
    instances: Iterable[Tuple[str, str]],
    penalties: Dict[str, Any],
    modes: Sequence[str] = DEFAULT_MODES,
    var_limit: Optional[int] = None,
    measure_full_horizon: bool = True,
) -> pd.DataFrame:
    """Long-format table, one row per (instance, problem, mode).

    `instances` is an iterable of (map_path, problem_name) pairs -- the same
    pairs a sweep config enumerates, so the two outputs join on
    (instance_map, problem_name, preprocess).
    """
    rows: List[Dict[str, Any]] = []
    for map_path, problem_name in instances:
        rows.extend(
            measure_instance(
                map_path,
                problem_name,
                penalties,
                modes,
                var_limit,
                measure_full_horizon,
            )
        )
    return pd.DataFrame(rows)


def instances_from_sweep_config(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Pull the (map, problem) pairs out of a sweep config so the scaling
    table covers exactly what the sweep runs -- and, unlike the sweep, also
    the instances its solvers cannot finish."""
    pairs: List[Tuple[str, str]] = []
    for entry in config.get("instances", []) or []:
        for problem_name in entry.get("problems", []) or []:
            pairs.append((entry["map"], problem_name))
    return pairs


def relative_to_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Add `vs_raw`: window_variables as a multiple of the same instance's
    `raw` count. This is the number the sweep cannot produce, because `raw` is
    the mode most likely to fail before recording anything."""
    df = df.copy()
    baseline = (
        df[df["preprocess"] == preprocess_modes.RAW]
        .set_index(["instance_map", "problem_name"])["window_variables"]
        .rename("_raw")
    )
    df = df.merge(baseline, on=["instance_map", "problem_name"], how="left")
    df["vs_raw"] = df["window_variables"] / df["_raw"]
    return df.drop(columns="_raw")


def relative_to_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Add `naive_gain`: how many times fewer variables this mode's whole
    windowed solve costs than the naive baseline -- no pruning and no
    windowing, one full-horizon QUBO.

    vs_raw isolates pruning and windowing_gain isolates windowing, but neither
    answers the question the pipeline exists to answer: against just writing
    the problem down and handing it over, what does all of this buy? That is
    the raw row's full_horizon_variables against each mode's
    total_window_variables, and it is the number that shows why raw is almost
    never the right choice -- as a windowed mode it is worse than the naive
    baseline it is derived from, because overlapping windows re-cover cells
    with no pruning to offset them.
    """
    df = df.copy()
    baseline = (
        df[df["preprocess"] == preprocess_modes.RAW]
        .set_index(["instance_map", "problem_name"])["full_horizon_variables"]
        .rename("_naive")
    )
    df = df.merge(baseline, on=["instance_map", "problem_name"], how="left")
    df["naive_gain"] = df["_naive"] / df["total_window_variables"].replace(0, pd.NA)
    return df.drop(columns="_naive")


def write_csv(df: pd.DataFrame, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)
