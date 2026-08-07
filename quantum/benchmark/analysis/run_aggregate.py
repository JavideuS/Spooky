"""
CLI wrapper for quantum.benchmark.analysis.aggregate.aggregate_sweep().

Usage:
  python -m quantum.benchmark.analysis.run_aggregate --sweep-dir results/sweeps/<sweep_id>
"""

import argparse
import sys

from quantum.benchmark.analysis.aggregate import aggregate_sweep


def main():
    parser = argparse.ArgumentParser(description="Aggregate a sweep's benchmark JSONs into CSVs.")
    parser.add_argument("--sweep-dir", "-d", required=True, help="Path to results/sweeps/<sweep_id>.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <sweep-dir>/analysis/.")
    args = parser.parse_args()

    try:
        tables = aggregate_sweep(args.sweep_dir, output_dir=args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[aggregate] {exc}", file=sys.stderr)
        sys.exit(1)

    for name, df in tables.items():
        print(f"[aggregate] {name}: {len(df)} rows")


if __name__ == "__main__":
    main()
