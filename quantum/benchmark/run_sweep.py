"""
CLI for running a benchmark sweep — see SweepRunner (sweep_runner.py) for
the actual matrix-execution logic.

Usage:
  spooky-sweep --config sweep_configs/smoke_test.yaml
  spooky-sweep --config sweep_configs/smoke_test.yaml --dry-run
  spooky-sweep --config sweep_configs/full_comparison.yaml \\
      --only-instances quantum/maps/synthetic/10x10/obs10x10_hard \\
      --only-solvers ilp,cbs
  spooky-sweep --config sweep_configs/full_comparison.yaml \\
      --enable-hardware --confirm-hardware-quota yes-spend-quota
  spooky-sweep --config sweep_configs/full_comparison.yaml --resume
  spooky-sweep --config sweep_configs/full_comparison.yaml \\
      --resume sweep_20260807_120000_ab12cd34

Run from the repo root — map paths in sweep configs are repo-root-relative
(e.g. "quantum/maps/synthetic/10x10/obs10x10_hard"), unlike qubo_cli.py's
single-solve paths which are relative to quantum/.
"""

import argparse
import sys

from quantum.benchmark.sweep_runner import (
    SweepRunner,
    SweepConfigError,
    HARDWARE_CONFIRM_PHRASE,
    AUTO_RESUME,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a benchmark sweep across a matrix of instances x solvers x ablations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to the sweep YAML config.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate the config and print the full execution plan without running any solver.",
    )
    parser.add_argument(
        "--only-instances", default=None, metavar="MAP1,MAP2",
        help="Comma-separated map paths to restrict the sweep to (matches the 'map' field exactly).",
    )
    parser.add_argument(
        "--only-solvers", default=None, metavar="NAME1,NAME2",
        help="Comma-separated solver entry names to restrict the sweep to (matches the 'name' field).",
    )
    parser.add_argument(
        "--enable-hardware", action="store_true",
        help="Allow hardware-gated solver entries (real quantum hardware) to run. "
             "Must be combined with --confirm-hardware-quota.",
    )
    parser.add_argument(
        "--confirm-hardware-quota", default="", metavar="PHRASE",
        help=f"Must exactly equal '{HARDWARE_CONFIRM_PHRASE}' together with --enable-hardware "
             f"for hardware-gated entries to actually run — two independent flags on purpose, "
             f"so a stray/copy-pasted command can't accidentally spend quota.",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=1, choices=[0, 1, 2, 3],
        help="Verbosity level forwarded to each solve (default: 1).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",       # 0 or 1 values: bare --resume auto-detects, --resume <id> is explicit
        const=AUTO_RESUME,  # value when --resume is given with no argument
        default=None,        # value when --resume is absent entirely
        metavar="SWEEP_ID",
        help="Resume a previous sweep. With no argument, automatically finds the most recently "
             "checkpointed incomplete sweep under the config's output_root. With an explicit id "
             "(printed at the start of every run, and part of its output_dir name), resumes that "
             "specific sweep. Any (instance, problem, solver, preprocess) combo that already has a "
             "completed benchmark JSON under that sweep's output_dir is skipped; everything else "
             "— including anything that previously failed or was interrupted — runs normally.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    only_instances = args.only_instances.split(",") if args.only_instances else None
    only_solvers = args.only_solvers.split(",") if args.only_solvers else None

    runner = SweepRunner(
        config_path=args.config,
        enable_hardware=args.enable_hardware,
        confirm_hardware=args.confirm_hardware_quota,
        dry_run=args.dry_run,
        only_instances=only_instances,
        only_solvers=only_solvers,
        verbose_level=args.verbose,
        resume_id=args.resume,
    )

    try:
        runner.load()
    except SweepConfigError as exc:
        print(f"[sweep] Invalid config: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.enable_hardware and not runner.enable_hardware:
        print(
            "[sweep] --enable-hardware was passed but --confirm-hardware-quota didn't match "
            f"'{HARDWARE_CONFIRM_PHRASE}' — hardware-gated entries will be skipped.",
            file=sys.stderr,
        )

    hardware_entries = [
        s["name"] for s in runner.config["solvers"] if s.get("hardware", False)
    ]
    if hardware_entries and runner.enable_hardware:
        print(
            f"[sweep] Hardware-gated entries ENABLED and will run, spending quota: "
            f"{hardware_entries}"
        )

    print(f"[sweep] sweep_id={runner.sweep_id}  output_dir={runner.output_dir}")
    index = runner.run()

    n_dry = sum(1 for e in index if e.get("dry_run"))
    n_resumed = sum(1 for e in index if e.get("resumed"))
    n_real = len(index) - n_dry - n_resumed
    n_hardware_skipped = len(runner.manifest["skipped"])
    n_failed = len(runner.manifest["failures"])
    print(
        f"[sweep] {'planned' if args.dry_run else 'ran'} {len(index)} combinations "
        f"({n_real} solved, {n_resumed} already done (resumed), {n_dry} dry-run only, "
        f"{n_hardware_skipped} hardware-skipped, {n_failed} failed)"
    )
    if n_failed:
        print(f"[sweep] See {runner.output_dir / 'manifest.json'} for failure details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
