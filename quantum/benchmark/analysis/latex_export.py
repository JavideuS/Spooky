"""
LaTeX table export for a sweep's aggregated results, generalizing the
column conventions of the paper, "Classical vs QUBO" comparison to
however many solvers a given sweep actually compared.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from quantum.benchmark.analysis.aggregate import _valid_mask


def export_benchmark_table(
    df: pd.DataFrame,
    output_path: str,
    caption: str = "Benchmark comparison across solvers",
    label: str = "tab:benchmark_results",
) -> str:
    """df must be the runs_long DataFrame (see quantum.benchmark.analysis.aggregate) —
    only execution_time_sec and average_reduction_ratio are used, both
    present without needing compute_energy_excess() to have run first.
    Writes the .tex snippet to output_path and returns it as a string, so
    it can be `\\input{}`'d into paper.tex or diffed against it."""
    valid = df[_valid_mask(df)]
    if valid.empty:
        raise ValueError("No valid runs to build a table from.")

    summary = (
        valid.groupby(["problem_name", "grid_size", "num_robots", "solver_name"])
        .agg(
            mean_time=("execution_time_sec", "mean"),
            mean_reduction=("average_reduction_ratio", "mean"),
        )
        .reset_index()
    )

    solver_names = sorted(summary["solver_name"].unique())
    index_cols = ["problem_name", "grid_size", "num_robots"]

    time_pivot = summary.pivot_table(
        index=index_cols, columns="solver_name", values="mean_time"
    )
    reduction_pivot = summary.pivot_table(
        index=index_cols, columns="solver_name", values="mean_reduction"
    )
    time_pivot.columns = [f"{c} Time (s)" for c in time_pivot.columns]
    reduction_pivot.columns = [f"{c} Reduction (\\%)" for c in reduction_pivot.columns]
    for c in reduction_pivot.columns:
        reduction_pivot[c] = (reduction_pivot[c] * 100).round(1)
    time_pivot = time_pivot.round(3)

    table = time_pivot.join(reduction_pivot).reset_index()
    table = table.rename(
        columns={"problem_name": "Problem", "grid_size": "Grid", "num_robots": "Robots"}
    )
    # Column order: identity columns, then all solvers' times, then all
    # solvers' reductions — grouped by kind rather than interleaved per
    # solver, easier to scan across a row.
    time_cols = [
        f"{s} Time (s)" for s in solver_names if f"{s} Time (s)" in table.columns
    ]
    reduction_cols = [
        f"{s} Reduction (\\%)"
        for s in solver_names
        if f"{s} Reduction (\\%)" in table.columns
    ]
    table = table[["Problem", "Grid", "Robots"] + time_cols + reduction_cols]

    column_format = "ll c" + "c" * (len(time_cols) + len(reduction_cols))
    latex = table.to_latex(
        index=False,
        column_format=column_format,
        escape=False,  # column names already have \% manually escaped
        na_rep="--",
        caption=caption,
        label=label,
        position="htbp",
        float_format="%.2f",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return latex
