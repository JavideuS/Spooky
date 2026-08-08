"""
Plotly plots for a sweep's aggregated results (see quantum.benchmark.analysis.aggregate).

Follows quantum/visualizer.py's conventions: its colorblind-safe categorical
palette (ROBOT_PALETTE, reused here for solvers), `hovertemplate='...
<extra></extra>'` on every trace, and a dual write_html()/write_image()
(kaleido) output pair.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quantum.visualizer import QuantumRoboticsVisualizer

_PALETTE = QuantumRoboticsVisualizer.ROBOT_PALETTE

# ---------------------------------------------------------------------------
# Shared theme applied to every figure via fig.update_layout(**_LAYOUT_THEME).
# Centralised here so all plots look consistent without repeating the same
# kwargs in every function. Deliberately excludes xaxis/yaxis: each plot
# passes its own xaxis=dict(**_AXIS_THEME, title=..., ...), and update_layout
# raises "got multiple values for keyword argument" if a key is supplied
# both via **_LAYOUT_THEME and explicitly.
# ---------------------------------------------------------------------------
_LAYOUT_THEME = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=13),
    title_font=dict(size=15, color="#1a1a2e"),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#cccccc",
        borderwidth=1,
    ),
    margin=dict(t=70, b=70, l=70, r=30),
)

# Merged into each plot's own xaxis=/yaxis= dict — never passed directly to
# update_layout (see _LAYOUT_THEME's note above).
_AXIS_THEME = dict(showgrid=True, gridcolor="#e8e8e8", zeroline=False)

# Bar marker border applied to every go.Bar trace.
_BAR_MARKER_LINE = dict(color="rgba(0,0,0,0.25)", width=0.8)


def _solver_colors(solver_names) -> Dict[str, str]:
    names = sorted(set(solver_names))
    return {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(names)}


def _save(fig: go.Figure, output_dir: str, name: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"{name}.html"
    fig.write_html(str(html_path))
    print(f"Plot saved to {html_path}")

    try:
        import kaleido  # noqa: F401
    except ImportError:
        print(f"Skipped PNG/PDF/SVG export for {name}: kaleido is not installed.")
        return

    # PNG for quick viewing; PDF/SVG are genuine vector output via kaleido
    # (not rasterized — verified directly, the SVG has no embedded raster
    # image), the right choice for camera-ready inclusion in a paper. PNG
    # alone is screen-DPI bitmap, wrong for print.
    for fmt in ("png", "pdf", "svg"):
        try:
            path = out / f"{name}.{fmt}"
            fig.write_image(str(path))
            print(f"{fmt.upper()} saved to {path}")
        except Exception as exc:
            # kaleido is confirmed present (checked above), so this is a
            # real per-format failure — show the actual cause, don't guess
            # "kaleido may be missing" when it demonstrably isn't.
            print(f"Skipped {fmt.upper()} export for {name}: {exc}")


def plot_scaling(
    df: pd.DataFrame, x: str = "num_robots", output_dir: Optional[str] = None, ci_z: float = 1.96
) -> go.Figure:
    """Mean wall-clock time vs. problem scale, one line per (solver,
    grid_size), with error bars showing an approximate 95% CI of the mean
    (mean +/- ci_z * std/sqrt(n), normal approximation — ci_z=1.96 is the
    95% default, reasonable once n isn't tiny). Deterministic solvers
    (ILP/CBS, num_runs=1) correctly get a zero-width bar rather than a
    fabricated one: std is NaN for a single-element group, filled to 0
    here since there's genuinely no run-to-run variability to report for
    those.

    The y-axis is log-scale, so the CI is drawn asymmetrically: the lower
    arm (arrayminus) is clamped to mean itself so mean-arrayminus never
    goes to/below zero (undefined on a log axis, and left unclamped Plotly
    draws it as if the axis were linear, which reads as a misleadingly
    truncated bar near small values). The upper arm is left as the plain
    symmetric CI."""
    valid = df[df["valid"]]
    agg = (
        valid.groupby(["grid_size", x, "solver_name"])["execution_time_sec"]
        .agg(mean="mean", std="std", n="size")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)
    agg["ci"] = ci_z * agg["std"] / np.sqrt(agg["n"])
    agg["ci_minus"] = np.minimum(agg["ci"], agg["mean"])
    colors = _solver_colors(agg["solver_name"])

    fig = go.Figure()
    for solver_name, solver_group in agg.groupby("solver_name"):
        for grid_size, sub in solver_group.groupby("grid_size"):
            sub = sub.sort_values(x)
            label = f"{solver_name} ({grid_size})"
            fig.add_trace(
                go.Scatter(
                    x=sub[x],
                    y=sub["mean"],
                    error_y=dict(
                        type="data",
                        array=sub["ci"],
                        arrayminus=sub["ci_minus"],
                        visible=True,
                        thickness=1.5,
                        width=4,
                    ),
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colors[solver_name], width=2.5),
                    marker=dict(size=9, color=colors[solver_name]),
                    customdata=sub["ci"],
                    hovertemplate=(
                        f"{label}<br>{x}=%{{x}}<br>mean=%{{y:.3f}}s"
                        "<br>95% CI=+/-%{customdata:.3f}s<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        **_LAYOUT_THEME,
        title="Solve time vs. problem scale (mean ± 95% CI)",
        xaxis=dict(
            **_AXIS_THEME,
            title=x,
            # Integer ticks only make sense while x defaults to num_robots;
            # skip forcing them if the caller passed a different column.
            **({"dtick": 1, "tickmode": "linear"} if x == "num_robots" else {}),
        ),
        yaxis=dict(
            **_AXIS_THEME,
            title="Mean execution time (s)",
            type="log",
        ),
    )
    if output_dir:
        _save(fig, output_dir, "scaling")
    return fig


def plot_success_rate_bars(df: pd.DataFrame, output_dir: Optional[str] = None) -> go.Figure:
    summary = (
        df.groupby(["instance_map", "problem_name", "solver_name"])["valid"]
        .mean()
        .reset_index()
    )
    summary["instance"] = summary["instance_map"].str.split("/").str[-1] + "/" + summary["problem_name"]
    colors = _solver_colors(summary["solver_name"])

    fig = go.Figure()
    for solver_name, group in summary.groupby("solver_name"):
        fig.add_trace(
            go.Bar(
                x=group["instance"],
                y=group["valid"],
                name=solver_name,
                marker=dict(
                    color=colors[solver_name],
                    line=_BAR_MARKER_LINE,
                ),
                hovertemplate=f"{solver_name}<br>%{{x}}<br>success rate=%{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_LAYOUT_THEME,
        title="Success rate by instance and solver",
        barmode="group",
        xaxis=dict(**_AXIS_THEME, title="Instance"),
        yaxis=dict(
            **_AXIS_THEME,
            title="Success rate",
            tickformat=".0%",
            range=[0, 1.05],      # explicit cap: full-success bars get breathing room
        ),
    )
    if output_dir:
        _save(fig, output_dir, "success_rate")
    return fig


def plot_optimality_gap(
    df: pd.DataFrame, output_dir: Optional[str] = None, boxpoints: str = "all"
) -> go.Figure:
    """Requires compute_optimality_gap() to have already run on df (adds
    the 'optimality_gap' column) — see quantum.benchmark.analysis.aggregate.

    boxpoints controls per-point overlay ("all", "outliers", "suspectedoutliers",
    or False to hide points entirely) — default "all" is fine for small
    sweeps but gets noisy with hundreds of runs per solver; pass "outliers"
    or False for those."""
    if "optimality_gap" not in df.columns:
        raise ValueError(
            "df has no 'optimality_gap' column — run compute_optimality_gap(df) first."
        )
    valid = df[df["valid"] & df["optimality_gap"].notna()]
    colors = _solver_colors(valid["solver_name"])

    fig = go.Figure()
    for solver_name, group in valid.groupby("solver_name"):
        fig.add_trace(
            go.Box(
                y=group["optimality_gap"],
                name=solver_name,
                marker=dict(color=colors[solver_name], size=5, opacity=0.6),
                line=dict(color=colors[solver_name]),
                boxpoints=boxpoints,
                jitter=0.4,           # spread raw points so they don't stack
                pointpos=-1.8,        # offset points to the left of the box
                hovertemplate=f"{solver_name}<br>gap=%{{y:.1%}}<extra></extra>",
            )
        )
    # Horizontal reference line at y=0: "as good as proven-optimal ILP"
    fig.add_hline(
        y=0,
        line=dict(color="#555555", dash="dash", width=1.2),
        annotation_text="ILP reference (0% gap)",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#555555"),
    )
    fig.update_layout(
        **_LAYOUT_THEME,
        title="Optimality gap distribution (vs. proven-optimal reference)",
        xaxis=dict(**_AXIS_THEME, title="Solver"),
        yaxis=dict(
            **_AXIS_THEME,
            title="Optimality gap",
            tickformat=".0%",
        ),
    )
    if output_dir:
        _save(fig, output_dir, "optimality_gap")
    return fig


def plot_path_efficiency(
    df: pd.DataFrame, output_dir: Optional[str] = None, boxpoints: str = "all"
) -> go.Figure:
    """Per-robot path efficiency (optimal length / moves actually taken —
    see quantum.benchmark.benchmark._compute_solution_statistics), one box
    per solver. df should be aggregate.load_robot_statistics()'s per-robot
    table, not runs_long (which only has the avg/min-per-run summary
    columns) — this plot wants the full per-robot distribution.

    Only validation_passed==True robots are plotted: efficiency computed
    off an invalid path (crashed into another robot, etc.) isn't a
    meaningful quality measure, same reasoning as plot_optimality_gap's
    valid-only filter."""
    valid = df[df["validation_passed"] & df["path_efficiency"].notna()]
    colors = _solver_colors(valid["solver_name"])

    fig = go.Figure()
    for solver_name, group in valid.groupby("solver_name"):
        fig.add_trace(
            go.Box(
                y=group["path_efficiency"],
                name=solver_name,
                marker=dict(color=colors[solver_name], size=5, opacity=0.6),
                line=dict(color=colors[solver_name]),
                boxpoints=boxpoints,
                jitter=0.4,
                pointpos=-1.8,
                hovertemplate=f"{solver_name}<br>efficiency=%{{y:.1%}}<extra></extra>",
            )
        )
    fig.add_hline(
        y=1.0,
        line=dict(color="#555555", dash="dash", width=1.2),
        annotation_text="Optimal (100%)",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#555555"),
    )
    fig.update_layout(
        **_LAYOUT_THEME,
        title="Per-robot path efficiency (optimal length / moves taken)",
        xaxis=dict(**_AXIS_THEME, title="Solver"),
        yaxis=dict(**_AXIS_THEME, title="Path efficiency", tickformat=".0%"),
    )
    if output_dir:
        _save(fig, output_dir, "path_efficiency")
    return fig


def plot_variable_reduction(df: pd.DataFrame, output_dir: Optional[str] = None) -> go.Figure:
    summary = (
        df[df["preprocess"] & df["average_reduction_ratio"].notna()]
        .groupby(["instance_map", "problem_name", "solver_name"])["average_reduction_ratio"]
        .mean()
        .reset_index()
    )
    summary["instance"] = summary["instance_map"].str.split("/").str[-1] + "/" + summary["problem_name"]
    colors = _solver_colors(summary["solver_name"])

    fig = go.Figure()
    for solver_name, group in summary.groupby("solver_name"):
        fig.add_trace(
            go.Bar(
                x=group["instance"],
                y=group["average_reduction_ratio"],
                name=solver_name,
                marker=dict(
                    color=colors[solver_name],
                    line=_BAR_MARKER_LINE,
                ),
                # Show mean % on top of each bar so the value is readable
                # without hovering — especially useful in static exports.
                text=[f"{v:.0%}" for v in group["average_reduction_ratio"]],
                textposition="outside",
                hovertemplate=f"{solver_name}<br>%{{x}}<br>reduction=%{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_LAYOUT_THEME,
        title="Variable/search-space reduction from BFS preprocessing",
        barmode="group",
        xaxis=dict(**_AXIS_THEME, title="Instance"),
        yaxis=dict(
            **_AXIS_THEME,
            title="Reduction ratio",
            tickformat=".0%",
            range=[0, 1.1],       # headroom for the text labels above bars
        ),
    )
    if output_dir:
        _save(fig, output_dir, "variable_reduction")
    return fig


def generate_all_plots(
    df: pd.DataFrame, output_dir: str, robot_df: Optional[pd.DataFrame] = None
) -> Dict[str, go.Figure]:
    """df should be aggregate.aggregate_sweep()'s runs_long (already has
    optimality_gap) for the full set of plots; plot_optimality_gap is
    skipped gracefully if that column is missing. robot_df is
    aggregate_sweep()'s robot_statistics_long — plot_path_efficiency is
    skipped gracefully if it's None or empty (e.g. sweep run at
    BenchmarkRunner level<2, which never records per-robot statistics)."""
    figs = {
        "scaling": plot_scaling(df, output_dir=output_dir),
        "success_rate": plot_success_rate_bars(df, output_dir=output_dir),
        "variable_reduction": plot_variable_reduction(df, output_dir=output_dir),
    }
    if "optimality_gap" in df.columns:
        figs["optimality_gap"] = plot_optimality_gap(df, output_dir=output_dir)
    if robot_df is not None and not robot_df.empty:
        figs["path_efficiency"] = plot_path_efficiency(robot_df, output_dir=output_dir)
    return figs
