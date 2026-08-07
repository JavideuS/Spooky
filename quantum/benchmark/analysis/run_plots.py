"""
CLI wrapper for quantum.benchmark.analysis.plots.generate_all_plots().

Usage:
  python -m quantum.benchmark.analysis.run_plots --sweep-dir results/sweeps/<sweep_id>
"""

import argparse
import sys

import pandas as pd

from quantum.benchmark.analysis.plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="Generate Plotly plots from a sweep's aggregated CSVs.")
    parser.add_argument("--sweep-dir", "-d", required=True, help="Path to results/sweeps/<sweep_id>.")
    parser.add_argument(
        "--runs-csv", default=None,
        help="Defaults to <sweep-dir>/analysis/runs_long.csv (run_aggregate.py first).",
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <sweep-dir>/analysis/plots/.")
    args = parser.parse_args()

    csv_path = args.runs_csv or f"{args.sweep_dir}/analysis/runs_long.csv"
    output_dir = args.output_dir or f"{args.sweep_dir}/analysis/plots"

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[plots] {csv_path} not found — run quantum.benchmark.analysis.run_aggregate first.", file=sys.stderr)
        sys.exit(1)

    figs = generate_all_plots(df, output_dir=output_dir)
    print(f"[plots] generated {len(figs)} plots in {output_dir}")


if __name__ == "__main__":
    main()
