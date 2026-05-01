"""
qubo_cli.py — Command-line interface for the Quantum QUBO Pathfinding solver.

This script exposes all runtime parameters via argparse, making it suitable
for deployment, scripted experiments, and integration with external systems.

For a hands-on Python example with inline comments and full control, see
qubo.py instead. This script mirrors its logic but driven entirely by CLI args.

Usage examples:
  # Basic solve with DWave
  python qubo_cli.py --map maps/synthetic/10x10/obs10x10_hard --problem four_robots --solver dwave

  # Benchmark run with PennyLane on GPU
  python qubo_cli.py --map maps/synthetic/5x5/obs5x5 --solver pennylane --device lightning.gpu --benchmark --num-runs 5

  # Graph-based problem
  python qubo_cli.py --map maps/graph/city --problem two_robots --builder graph --solver dwave --penalty-set graph

  # Override penalties individually
  python qubo_cli.py --map maps/synthetic/10x10/obs10x10_hard --K-hot 9 --K-adj 4.8 --K-start 6.5 --K-goal 3.0

  # Suppress all output (silent mode)
  python qubo_cli.py --map maps/synthetic/10x10/obs10x10_hard --verbose 0
"""

import argparse
import sys

from pennylane import numpy as np
from solvers import SolverFactory
from pathFormulation import PathfindingProblem
import config.parser as config_parser
from builder import QUBOBuilder, GraphQUBO
import benchmark.benchmark as bm_module
from quantum.utils.logger import set_verbose_level, get_logger
import time


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qubo_cli",
        description=(
            "Quantum QUBO Pathfinding solver CLI.\n\n"
            "For a fully-annotated Python example, see qubo.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Problem definition ------------------------------------------------
    prob = parser.add_argument_group("Problem")
    prob.add_argument(
        "--map", "-m",
        required=True,
        metavar="PATH",
        help="Path (without extension) to the map config, e.g. maps/synthetic/10x10/obs10x10_hard",
    )
    prob.add_argument(
        "--problem", "-p",
        default="four_robots",
        metavar="NAME",
        help="Problem name defined inside the map config (default: four_robots)",
    )
    prob.add_argument(
        "--builder", "-b",
        choices=["grid", "graph"],
        default="grid",
        help="QUBO builder type: 'grid' (QUBOBuilder) or 'graph' (GraphQUBO) (default: grid)",
    )
    prob.add_argument(
        "--distance-scaling",
        default="enhanced_linear",
        metavar="MODE",
        help="Distance scaling mode passed to QUBOBuilder (default: enhanced_linear)",
    )
    prob.add_argument(
        "--window-limit",
        default=[],
        nargs="+",
        metavar="ROBOT=N",
        help=(
            "Per-robot window limits as 'robot_id=N' pairs, e.g. --window-limit robot_0=5 robot_1=3. "
            "See qubo.py for a single-robot example."
        ),
    )
    prob.add_argument(
        "--var-limit",
        type=int,
        default=None,
        metavar="N",
        help="Variable limit passed to QUBO builders. (default: Grid=1650, Graph=1201)",
    )

    # ---- Penalty set -------------------------------------------------------
    pen = parser.add_argument_group("Penalties")
    pen.add_argument(
        "--penalty-set",
        default="crash",
        metavar="SET",
        help=(
            "Named penalty set from config.yaml to use as base "
            "(default: crash). Overridden by individual --K-* flags."
        ),
    )
    # Individual overrides — if given, they take precedence over the set
    pen.add_argument("--K-hot",   type=float, default=None, metavar="VAL", help="Override K_hot penalty")
    pen.add_argument("--K-adj",   type=float, default=None, metavar="VAL", help="Override K_adj penalty")
    pen.add_argument("--K-start", type=float, default=None, metavar="VAL", help="Override K_start penalty")
    pen.add_argument("--K-goal",  type=float, default=None, metavar="VAL", help="Override K_goal penalty")
    pen.add_argument("--K-lock",  type=float, default=None, metavar="VAL", help="Override K_lock penalty")
    pen.add_argument("--K-bt",    type=float, default=None, metavar="VAL", help="Override K_bt penalty")
    pen.add_argument("--K-tp",    type=float, default=None, metavar="VAL", help="Override K_tp penalty")
    pen.add_argument("--K-crash", type=float, default=None, metavar="VAL", help="Override K_crash penalty")
    pen.add_argument("--K-obs",   type=float, default=None, metavar="VAL", help="Override K_obs penalty")
    pen.add_argument("--K-goal-approx", type=float, default=None, metavar="VAL", help="Override K_goal_approx penalty")

    # ---- Solver ------------------------------------------------------------
    sol = parser.add_argument_group("Solver")
    sol.add_argument(
        "--solver", "-s",
        choices=["dwave", "pennylane", "qiskit_remote"],
        default="dwave",
        help="Solver backend (default: dwave)",
    )
    sol.add_argument(
        "--normalize-scale",
        type=float,
        default=None,
        metavar="N",
        help=(
            "QUBO normalization scale factor. "
            "Defaults: dwave=4.0, pennylane=1.0. "
            "See qubo.py comments for per-qubit guidance."
        ),
    )
    sol.add_argument(
        "--num-reads",
        default=None,
        metavar="N|auto",
        help=(
            "Number of solver reads. Pass an integer or 'auto' "
            "(default: dwave=4, pennylane=auto)"
        ),
    )

    # PennyLane / QAOA-specific
    pl = parser.add_argument_group("PennyLane / QAOA (only used when --solver pennylane)")
    pl.add_argument(
        "--device",
        default="lightning.gpu",
        metavar="DEV",
        help=(
            "PennyLane device string, e.g. 'lightning.gpu', 'lightning.qubit', "
            "'qiskit.remote' (default: lightning.gpu)"
        ),
    )
    pl.add_argument(
        "--layers",
        type=int,
        default=2,
        metavar="N",
        help="Number of QAOA layers (default: 2)",
    )
    pl.add_argument(
        "--optimizer",
        default="QNG",
        metavar="OPT",
        help="Optimizer name passed to PennyLane solver (default: QNG)",
    )
    pl.add_argument(
        "--opt-steps",
        type=int,
        default=30,
        metavar="N",
        help="Number of optimizer steps (default: 30)",
    )
    pl.add_argument(
        "--init-params",
        default=None,
        metavar="FILE",
        help=(
            "Path to a .npy file containing initial QAOA parameters. "
            "If omitted, the built-in default params from qubo.py are used."
        ),
    )

    # ---- Run mode ----------------------------------------------------------
    run = parser.add_argument_group("Run mode")
    run_ex = run.add_mutually_exclusive_group()
    run_ex.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run in benchmark mode (multiple runs, saves JSON results with path validation). "
            "Use --num-runs 1 as a single validated solve with path checking instead of bare --solve."
        ),
    )
    run_ex.add_argument(
        "--solve",
        action="store_true",
        default=True,
        help="[default] Run a single solve and print the decoded path",
    )
    run.add_argument(
        "--num-runs",
        type=int,
        default=10,
        metavar="N",
        help="Number of benchmark runs (only used with --benchmark, default: 10)",
    )
    run.add_argument(
        "--benchmark-level",
        type=int,
        choices=[1, 2, 3],
        default=2,
        metavar="1|2|3",
        help=(
            "Benchmark output detail level: "
            "1=Summary only, 2=+Paths, 3=+Raw bits (default: 2)"
        ),
    )

    # ---- Config & misc -----------------------------------------------------
    misc = parser.add_argument_group("Config & misc")
    misc.add_argument(
        "--config",
        default="config/config.yaml",
        metavar="FILE",
        help="Path to the main YAML config file (default: config/config.yaml)",
    )
    misc.add_argument(
        "--materials",
        default="config/materials.yaml",
        metavar="FILE",
        help="Path to the materials YAML file (default: config/materials.yaml)",
    )
    misc.add_argument(
        "--verbose", "-v",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        metavar="0-3",
        help=(
            "Verbose level: 0=Silent, 1=Minimal, 2=Standard, 3=Debug. "
            "Overrides the value in config.yaml."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_window_limits(raw: list[str]) -> dict:
    """Parse ['robot_0=5', 'robot_1=3'] → {'robot_0': 5, 'robot_1': 3}."""
    limits = {}
    for entry in raw:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"Invalid --window-limit format '{entry}'. Expected 'robot_id=N'."
            )
        robot_id, n = entry.split("=", 1)
        limits[robot_id.strip()] = int(n.strip())
    return limits


def build_penalties(config: dict, args: argparse.Namespace) -> dict:
    """
    Start from the named penalty set in config.yaml, then apply any individual
    --K-* overrides supplied via the CLI.
    """
    penalties = dict(config["penalty_sets"][args.penalty_set])
    penalties.setdefault("name", args.penalty_set)

    overrides = {
        "K_hot":        args.K_hot,
        "K_adj":        args.K_adj,
        "K_start":      args.K_start,
        "K_goal":       args.K_goal,
        "K_lock":       args.K_lock,
        "K_bt":         args.K_bt,
        "K_tp":         args.K_tp,
        "K_crash":      args.K_crash,
        "K_obs":        args.K_obs,
        "K_goal_approx": args.K_goal_approx,
    }
    for key, val in overrides.items():
        if val is not None:
            penalties[key] = val

    return penalties


def build_solver(args: argparse.Namespace, verbose_level: int):
    """Instantiate the correct solver from CLI arguments."""
    logger = get_logger()

    if args.solver == "dwave":
        norm_scale = args.normalize_scale if args.normalize_scale is not None else 4.0
        num_reads  = int(args.num_reads) if args.num_reads and args.num_reads != "auto" else 4
        logger.minimal(f"Creating DWave solver (scale={norm_scale}, reads={num_reads})")
        return SolverFactory.create_solver(
            solver="dwave",
            normalize_scale=norm_scale,
            num_reads=num_reads,
        )

    elif args.solver == "pennylane":
        norm_scale = args.normalize_scale if args.normalize_scale is not None else 1.0
        num_reads  = args.num_reads if args.num_reads else "auto"

        # Initial QAOA params — load from file or use the defaults from qubo.py
        if args.init_params:
            init_params = np.load(args.init_params, allow_pickle=False)
            init_params = np.array(init_params, requires_grad=True)
            logger.minimal(f"Loaded init_params from {args.init_params}")
        else:
            # Default params tuned for 2-layer QAOA (see qubo.py for context)
            init_params = np.array(
                [[1.70579, 0.70321062], [0.49879231, 0.49412656]],
                requires_grad=True,
            )

        logger.minimal(
            f"Creating PennyLane solver (device={args.device}, "
            f"layers={args.layers}, optimizer={args.optimizer}, "
            f"steps={args.opt_steps}, scale={norm_scale})"
        )
        return SolverFactory.create_solver(
            solver="pennylane",
            normalize_scale=norm_scale,
            num_reads=num_reads,
            layers=args.layers,
            optimizer=args.optimizer,
            opt_steps=args.opt_steps,
            device=args.device,
            params=init_params,
            verbose_level=verbose_level,
        )

    elif args.solver == "qiskit_remote":
        # Mirrors the qiskit_hardware setup in qubo.py:
        #   SolverFactory.create_solver(solver="pennylane", device="qiskit.remote", ...)
        # normalize_scale defaults to 4.0 (same as qubo.py's qiskit_hardware).
        norm_scale = args.normalize_scale if args.normalize_scale is not None else 4.0
        num_reads  = args.num_reads if args.num_reads else "auto"
        device     = args.device if args.device != "lightning.gpu" else "qiskit.remote"

        if args.init_params:
            init_params = np.load(args.init_params, allow_pickle=False)
            init_params = np.array(init_params, requires_grad=True)
            logger.minimal(f"Loaded init_params from {args.init_params}")
        else:
            init_params = np.array(
                [[1.70579, 0.70321062], [0.49879231, 0.49412656]],
                requires_grad=True,
            )

        logger.minimal(
            f"Creating Qiskit-remote solver via PennyLane "
            f"(device={device}, layers={args.layers}, "
            f"optimizer={args.optimizer}, steps={args.opt_steps}, scale={norm_scale})"
        )
        return SolverFactory.create_solver(
            solver="pennylane",
            normalize_scale=norm_scale,
            num_reads=num_reads,
            layers=args.layers,
            optimizer=args.optimizer,
            opt_steps=args.opt_steps,
            device=device,
            params=init_params,
            verbose_level=verbose_level,
        )

    else:
        raise ValueError(f"Unknown solver: {args.solver}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # -- Config --------------------------------------------------------------
    config = config_parser.load_config(args.config, sections=["penalty_sets", "verbose"])

    # Verbose: CLI > config.yaml
    verbose_level = args.verbose if args.verbose is not None else config["verbose"]["level"]
    set_verbose_level(verbose_level)
    logger = get_logger()

    logger.minimal(f"qubo_cli starting | map={args.map} | problem={args.problem}")

    # -- Materials -----------------------------------------------------------
    materials_data = config_parser.load_config(args.materials)["materials"]

    # -- Problem -------------------------------------------------------------
    problem = PathfindingProblem.from_map_config(
        args.map,
        problem_name=args.problem,
        materials_data=materials_data,
    )

    # -- Penalties -----------------------------------------------------------
    if args.penalty_set not in config["penalty_sets"]:
        logger.minimal(
            f"[ERROR] Penalty set '{args.penalty_set}' not found in {args.config}. "
            f"Available: {list(config['penalty_sets'].keys())}"
        )
        sys.exit(1)

    penalties = build_penalties(config, args)
    logger.minimal(f"Using penalty set: {args.penalty_set} | effective: {penalties}")

    # -- Window limits -------------------------------------------------------
    window_limits = parse_window_limits(args.window_limit)

    # -- Builder -------------------------------------------------------------
    builder_kwargs = {
        "penalties": penalties,
        "name": args.problem,
        "robot_window_limits": window_limits if window_limits else None,
    }
    if args.var_limit is not None:
        builder_kwargs["var_limit"] = args.var_limit

    if args.builder == "grid":
        p = problem.as_grid_only()
        builder_kwargs["distance_scaling"] = args.distance_scaling
        builder = QUBOBuilder(p, **builder_kwargs)
    else:  # graph
        p = problem.as_graph_only()
        builder = GraphQUBO(p, **builder_kwargs)

    logger.minimal(f"Builder: {args.builder.upper()} | window_limits={window_limits or 'none'}")

    # -- Solver --------------------------------------------------------------
    solver = build_solver(args, verbose_level)

    # -- Run mode ------------------------------------------------------------
    if args.benchmark:
        logger.minimal(f"Running benchmark: {args.num_runs} runs, level {args.benchmark_level}")
        runner = bm_module.BenchmarkRunner(
            builder,
            solver,
            num_runs=args.num_runs,
            level=args.benchmark_level,
        )
        runner.run_build()

    else:
        # Single solve
        timer = time.time()
        builder.build()
        solution = solver.solve_qubo_smart(builder, False)
        path = solver.decode_path(solution["solution"], problem)

        energy = solution["energy"]
        if isinstance(energy, list):
            energy = sum(energy)

        logger.debug(f"Raw path:  {path}")  # Full decoded tuples — only useful at verbose=3
        logger.minimal(f"Energy:    {energy:.4f}")
        logger.minimal(f"Time:      {time.time() - timer:.4f}")

        for robot_id, robot in problem.robots.items():
            logger.minimal(f"  [{robot_id}] {robot.path}")


if __name__ == "__main__":
    main()
