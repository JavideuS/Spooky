"""
Convenience CLI chaining aggregate -> plots -> LaTeX export for a sweep.
Every step is independently importable/callable too (aggregate_sweep,
generate_all_plots, export_benchmark_table)
This is meant for FastAPI calling the underlying functions directly without
needing this CLI.

Usage:
  python -m quantum.benchmark.analysis.run_report --sweep-dir results/sweeps/<sweep_id>
"""

import argparse
import sys

from quantum.benchmark.analysis.aggregate import aggregate_sweep
from quantum.benchmark.analysis.plots import generate_all_plots
from quantum.benchmark.analysis.latex_export import export_benchmark_table


def main():
    parser = argparse.ArgumentParser(
        description="Run the full aggregate -> plots -> LaTeX pipeline for a sweep."
    )
    parser.add_argument(
        "--sweep-dir", "-d", required=True, help="Path to results/sweeps/<sweep_id>."
    )
    parser.add_argument(
        "--caption",
        default="Benchmark comparison across solvers",
        help="LaTeX table caption.",
    )
    args = parser.parse_args()

    try:
        tables = aggregate_sweep(args.sweep_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[report] {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"[report] aggregated {len(tables['runs_long'])} runs")

    plots_dir = f"{args.sweep_dir}/analysis/plots"
    figs = generate_all_plots(tables["runs_long"], output_dir=plots_dir)
    print(f"[report] generated {len(figs)} plots in {plots_dir}")

    table_path = f"{args.sweep_dir}/analysis/benchmark_table.tex"
    export_benchmark_table(tables["runs_long"], table_path, caption=args.caption)
    print(f"[report] wrote LaTeX table to {table_path}")


if __name__ == "__main__":
    main()
