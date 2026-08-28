"""
CLI wrapper for quantum.benchmark.analysis.variable_scaling.

Measures QUBO size per pre-processing mode without solving anything, so it
covers instances no backend can finish -- including `raw`, the unpruned-window
baseline every reduction ratio in a sweep is implicitly measured against.
Note `raw` is still windowed: it is the pruning baseline, not the problem
size. encoded_variables is the problem size.

Usage:
  python -m quantum.benchmark.analysis.run_variable_scaling \
      --config sweep_configs/classical_test.yaml -o results/variable_scaling.csv

  python -m quantum.benchmark.analysis.run_variable_scaling \
      --map quantum/maps/synthetic/10x10/obs10x10_hard --problem four_robots

  python -m quantum.benchmark.analysis.run_variable_scaling \
      --sweep-dir results/sweeps/<sweep_id>
"""

import argparse
import sys
from pathlib import Path

import yaml

from quantum.benchmark.analysis.variable_scaling import (
    DEFAULT_MODES,
    instances_from_sweep_config,
    measure_scaling,
    penalty_set_from_config,
    relative_to_naive,
    relative_to_raw,
    sweep_config_from_dir,
    write_csv,
)
from quantum.config import parser as config_parser
from quantum.utils import preprocess as preprocess_modes
from quantum.utils.logger import set_verbose_level

_CONFIG_YAML = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Measure QUBO variable counts per pre-processing mode (no solving)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config", help="Sweep config whose instances/problems to measure."
    )
    source.add_argument(
        "--sweep-dir",
        "-d",
        help=(
            "Finished sweep whose instances to measure, read from its "
            "manifest.json. Defaults the penalty set to the one that sweep's "
            "QUBO solvers used, and writes the CSV into its analysis/ "
            "directory -- same addressing as run_aggregate and run_plots."
        ),
    )
    source.add_argument("--map", help="Single map path (use with --problem).")
    parser.add_argument("--problem", help="Problem name, required with --map.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        choices=list(preprocess_modes.MODES),
        help=f"Defaults to {list(DEFAULT_MODES)}.",
    )
    parser.add_argument(
        "--penalty-set",
        default=None,
        help=(
            "Penalty set name from config.yaml. Defaults to the sweep's own "
            "with --sweep-dir, otherwise 'crash'."
        ),
    )
    parser.add_argument(
        "--var-limit",
        type=int,
        default=None,
        help="Builder var_limit; defaults to the builder's own default.",
    )
    parser.add_argument(
        "--no-full-horizon",
        action="store_true",
        help=(
            "Skip the un-windowed QUBO build used for windowing_gain. It is "
            "cheap up to 10x10 but is the one measurement that scales with "
            "the map, so turn it off for very large grids."
        ),
    )
    parser.add_argument("--output", "-o", default=None, help="Write a CSV here.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help=(
            "Global log level. Defaults to 0: the builders log their window "
            "sizing on construction, and this tool builds every instance x "
            "mode, so anything higher buries the table."
        ),
    )
    args = parser.parse_args()

    if args.map and not args.problem:
        parser.error("--map requires --problem")

    # the logger is a process-wide singleton, so the builders' own
    # verbose_level=0 does not silence it -- this does
    set_verbose_level(args.verbose)

    penalty_set = args.penalty_set
    if args.sweep_dir:
        config = sweep_config_from_dir(args.sweep_dir)
        instances = instances_from_sweep_config(config)
        penalty_set = penalty_set or penalty_set_from_config(config)
    elif args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            instances = instances_from_sweep_config(yaml.safe_load(f))
    else:
        instances = [(args.map, args.problem)]
    penalty_set = penalty_set or "crash"

    penalty_sets = config_parser.load_config(
        str(_CONFIG_YAML), sections=["penalty_sets"]
    ).get("penalty_sets", {})
    if penalty_set not in penalty_sets:
        print(
            f"[scaling] penalty set '{penalty_set}' not in {_CONFIG_YAML}. "
            f"Available: {sorted(penalty_sets)}",
            file=sys.stderr,
        )
        sys.exit(1)

    df = measure_scaling(
        instances,
        penalty_sets[penalty_set],
        args.modes,
        args.var_limit,
        measure_full_horizon=not args.no_full_horizon,
    )
    if df.empty:
        print("[scaling] nothing to measure", file=sys.stderr)
        sys.exit(1)

    df = relative_to_naive(relative_to_raw(df))
    failed = df[df["error"].notna()]
    df = df[df["error"].isna()]

    columns = [
        "instance_map",
        "problem_name",
        "preprocess",
        "num_robots",
        "horizon",
        "encoded_variables",
        "num_windows",
        "window_variables",
        "window_variables_after_numeric",
        "total_window_variables",
        "full_horizon_variables",
        "windowing_gain",
        "vs_raw",
        "naive_gain",
    ]
    shown = df[columns].copy()
    shown["instance_map"] = shown["instance_map"].str.split("/").str[-1]
    print(shown.to_string(index=False))

    for _, row in failed.iterrows():
        print(
            f"[scaling] {row['instance_map']}/{row['problem_name']} "
            f"({row['preprocess']}): {row['error']}",
            file=sys.stderr,
        )

    output = args.output
    if output is None and args.sweep_dir:
        output = str(Path(args.sweep_dir) / "analysis" / "variable_scaling.csv")
    if output:
        print(f"[scaling] wrote {write_csv(df, output)}")


if __name__ == "__main__":
    main()
