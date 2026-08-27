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

  # Benchmark run with PennyLane on Windows (no lightning.gpu wheels; use lightning.qubit)
  python qubo_cli.py --map maps/synthetic/5x5/obs5x5 --solver pennylane --device lightning.qubit --benchmark --num-runs 5

  # Graph-based problem
  python qubo_cli.py --map maps/graph/city --problem two_robots --builder graph --solver dwave --penalty-set graph

  # Solve directly against a bare .h5 (no companion problems YAML needed)
  python qubo_cli.py --map maps/synthetic/fraunhofer/test-scenario_simple --robot robot_0=7,5:7,49 --solver dwave

  # Single-robot shorthand, real-world (ROS map frame) coordinates in meters
  python qubo_cli.py --map maps/synthetic/fraunhofer/test-scenario_simple --coordinate-format world --start -9.8,17.8 --goal -9.4,17.8 --solver dwave

  # Override penalties individually
  python qubo_cli.py --map maps/synthetic/10x10/obs10x10_hard --K-hot 9 --K-adj 4.8 --K-start 6.5 --K-goal 3.0

  # Suppress all output (silent mode)
  python qubo_cli.py --map maps/synthetic/10x10/obs10x10_hard --verbose 0

  # Solve and open an animated visualization in the browser
  python qubo_cli.py --map maps/synthetic/5x5/obs5x5 --problem two_robots --visualize

  # Solve and save the animation as a GIF (or .html for interactive)
  python qubo_cli.py --map maps/synthetic/5x5/obs5x5 --problem two_robots --visualize -o run.gif
"""

import argparse
import sys
from pathlib import Path

from pennylane import numpy as np
from quantum.solvers import SolverFactory
from quantum.utils import preprocess as preprocess_modes
from quantum.pathFormulation import PathfindingProblem, InfeasibleProblemError
import quantum.config.parser as config_parser
from quantum.builder import (
    QUBOBuilder, GraphQUBO, GridILPBuilder, GraphILPBuilder,
    GridCBSBuilder, GraphCBSBuilder,
)
import quantum.benchmark.benchmark as bm_module
from quantum.utils.logger import set_verbose_level, get_logger
from quantum.utils.paths import clip_path_at_goal
import time

_HERE = Path(__file__).parent  # quantum/


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _resolve_preprocess(args) -> str:
    """--preprocess wins; --no-preprocess is kept as an alias for 'raw'.

    Both are accepted so existing scripts and docs keep working. Passing them
    contradictorily is an error rather than a silent precedence rule."""
    mode = getattr(args, "preprocess", None)
    if mode is None:
        return preprocess_modes.RAW if args.no_preprocess else preprocess_modes.FULL
    if args.no_preprocess and mode != preprocess_modes.RAW:
        raise SystemExit(
            f"--no-preprocess conflicts with --preprocess {mode}. "
            "--no-preprocess is an alias for --preprocess raw; pass only one."
        )
    return preprocess_modes.normalize(mode)


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
        "--map",
        "-m",
        required=True,
        metavar="PATH",
        help="Path (without extension) to the map config, e.g. maps/synthetic/10x10/obs10x10_hard",
    )
    prob.add_argument(
        "--problem",
        "-p",
        default="four_robots",
        metavar="NAME",
        help=(
            "Problem name defined inside the map config (default: four_robots). "
            "Ignored when --robot or --start/--goal are given."
        ),
    )
    prob.add_argument(
        "--builder",
        "-b",
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
        metavar="N or ROBOT=N",
        help=(
            "Window step limit. Pass a single integer to cap all robots globally "
            "(e.g. --window-limit 6), or 'robot_id=N' pairs for per-robot limits "
            "(e.g. --window-limit robot_0=5 robot_1=3)."
        ),
    )
    prob.add_argument(
        "--var-limit",
        type=int,
        default=None,
        metavar="N",
        help="Variable limit passed to QUBO builders. (default: Grid=1650, Graph=1201)",
    )
    prob.add_argument(
        "--coordinate-format",
        choices=["matrix", "cartesian", "world"],
        default="matrix",
        help=(
            "Coordinate convention for start/goal in the problem config and for "
            "printed/visualized output paths: 'matrix' (row, col), Spooky's native "
            "convention (default), 'cartesian' (x, y) robotics/Y-up, or 'world' "
            "(real-world meters, ROS map frame -- requires the map .h5 to carry "
            "an origin/resolution, see quantum/maps/pgm2HDF5.py). Also governs how "
            "--robot/--start/--goal tokens are interpreted. Per-robot "
            "'coordinate_format' entries in the map YAML take precedence over this."
        ),
    )
    prob.add_argument(
        "--robot",
        action="append",
        default=[],
        metavar="ID=START:GOAL[:PRIORITY[:SAFETY_RADIUS[:START_TIME]]]",
        help=(
            "Define a robot directly on the command line instead of looking it up "
            "in a problems YAML -- solves straight against --map's .h5, no "
            "companion .yaml required. Repeatable for multiple robots. START/GOAL "
            "are each 'a,b', interpreted per --coordinate-format. "
            "PRIORITY/SAFETY_RADIUS/START_TIME are optional (defaults: 1.0, 0.5, 0). "
            "Example: --robot robot_0=0,0:9,9 --robot robot_1=5,5:0,0:2.0. "
            "Mutually exclusive with --problem/--start/--goal."
        ),
    )
    prob.add_argument(
        "--start",
        default=None,
        metavar="ROW,COL or X,Y",
        help=(
            "Shorthand for a single robot's start position, paired with --goal -- "
            "equivalent to '--robot robot_0=START:GOAL'. Mutually exclusive with --robot."
        ),
    )
    prob.add_argument(
        "--goal",
        default=None,
        metavar="ROW,COL or X,Y",
        help="Shorthand for a single robot's goal position, paired with --start.",
    )
    prob.add_argument(
        "--horizon",
        "-T",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Time horizon override (only used with --robot/--start+--goal; the "
            "problem YAML's own time_limit is used otherwise). If omitted, each "
            "robot's horizon is computed automatically from its path length."
        ),
    )
    prob.add_argument(
        "--no-reduction-log",
        action="store_true",
        default=False,
        help=(
            "Disable reduction logging during QUBO preprocessing. "
            "Faster, but prevents BFS-based variable unfixing during diagonal reduction. "
            "Use to benchmark the overhead of reduction tracking."
        ),
    )

    # ---- Penalty set -------------------------------------------------------
    pen = parser.add_argument_group("Penalties")
    pen.add_argument(
        "--penalty-set",
        default="swap",
        metavar="SET",
        help=(
            "Named penalty set from config.yaml to use as base "
            "(default: swap). Overridden by individual --K-* flags."
        ),
    )
    # Individual overrides — if given, they take precedence over the set
    pen.add_argument(
        "--K-hot",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_hot penalty",
    )
    pen.add_argument(
        "--K-adj",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_adj penalty",
    )
    pen.add_argument(
        "--K-start",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_start penalty",
    )
    pen.add_argument(
        "--K-goal",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_goal penalty",
    )
    pen.add_argument(
        "--K-lock",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_lock penalty",
    )
    pen.add_argument(
        "--K-bt", type=float, default=None, metavar="VAL", help="Override K_bt penalty"
    )
    pen.add_argument(
        "--K-tp", type=float, default=None, metavar="VAL", help="Override K_tp penalty"
    )
    pen.add_argument(
        "--K-crash",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_crash penalty",
    )
    pen.add_argument(
        "--K-swap",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_swap penalty",
    )
    pen.add_argument(
        "--K-obs",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_obs penalty",
    )
    pen.add_argument(
        "--K-goal-approx",
        type=float,
        default=None,
        metavar="VAL",
        help="Override K_goal_approx penalty",
    )

    # ---- Solver ------------------------------------------------------------
    sol = parser.add_argument_group("Solver")
    sol.add_argument(
        "--solver",
        "-s",
        choices=["dwave", "pennylane", "qiskit_remote", "qiskit_iqm", "ilp", "cbs"],
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
    sol.add_argument(
        "--preprocess",
        choices=list(preprocess_modes.MODES),
        default=None,
        help=(
            "Pre-processing mode. 'full' (default) is aggressive BFS pruning "
            "plus numerical variable fixing; 'bfs_aggressive' and 'bfs_safe' "
            "keep the windowed loop and the chosen BFS but skip the numerical "
            "stage; 'full_safe' is monotone BFS plus numerical; 'raw' is the "
            "old --no-preprocess simple loop. bfs_safe/full_safe are the only "
            "modes whose pruning cannot exclude a feasible solution, at "
            "roughly 10-19x the variables. See quantum/utils/preprocess.py."
        ),
    )
    sol.add_argument(
        "--no-preprocess",
        action="store_true",
        default=False,
        help=(
            "Disable variable reduction (QUBO: BFS logical-variable reduction, "
            "diagonal pruning, and the correction loop, runs the simple "
            "raw-sampler loop instead. ILP: BFS reachability pruning of the "
            "decision variables, solves the unpruned model instead). Enabled "
            "by default for both."
        ),
    )
    sol.add_argument(
        "--pyomo-solver",
        default="appsi_highs",
        metavar="NAME",
        help=(
            "Pyomo solver backend name (only used with --solver ilp), "
            "e.g. 'appsi_highs' (default), 'cbc', 'glpk'."
        ),
    )
    sol.add_argument(
        "--time-limit",
        type=float,
        default=30,
        metavar="SECONDS",
        help=(
            "Solver time limit in seconds (--solver ilp or cbs). ILP/HiGHS "
            "returns its best incumbent found so far if the limit is hit "
            "before proving optimality; CBS returns its best incumbent "
            "found in its constraint-tree search so far. Default: 30."
        ),
    )
    sol.add_argument(
        "--node-limit",
        type=int,
        default=5000,
        metavar="N",
        help=(
            "Max constraint-tree nodes CBS will expand (only used with "
            "--solver cbs) — a second safety cap alongside --time-limit, "
            "whichever is hit first stops the search. Default: 5000."
        ),
    )

    # PennyLane / QAOA-specific
    pl = parser.add_argument_group(
        "PennyLane / QAOA (only used when --solver pennylane)"
    )
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
        "--threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "CPU thread count for lightning.qubit's OpenMP backend "
            "(default.qubit/lightning.gpu unaffected). Default: unset, uses "
            "every available core (OpenMP's own default)."
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
    pl.add_argument(
        "--machine",
        default=None,
        metavar="NAME",
        help=(
            "Pin a specific hardware backend/machine (only used with "
            "--device qiskit.remote or qiskit.iqm / --solver qiskit_remote or "
            "qiskit_iqm). IBM: an exact backend name, e.g. 'ibm_torino' "
            "(default: least_busy). IQM: 'sirius', 'garnet', or 'emerald' "
            "(default: auto-picks the smallest that fits the window)."
        ),
    )
    pl.add_argument(
        "--no-session",
        action="store_true",
        default=False,
        help=(
            "Only used with --device qiskit.remote / --solver qiskit_remote. "
            "Disable holding one IBM Runtime Session open across the whole "
            "windowed solve — every window's job goes back through the "
            "public queue instead. Session is on by default and requires a "
            "plan that supports it (IBM's Open plan doesn't; Pay-As-You-Go "
            "and above do) — it already falls back automatically with a "
            "logged warning if opening one fails, so --no-session is rarely "
            "needed, mainly for isolating whether Session itself is the "
            "cause of an issue."
        ),
    )
    pl.add_argument(
        "--session-max-time",
        type=int,
        default=None,
        metavar="SECONDS",
        help=(
            "Only used with --device qiskit.remote / --solver qiskit_remote "
            "(and --no-session is not set). Max seconds the IBM Runtime "
            "Session stays open. Default: unset, uses IBM's own default "
            "(900s at time of writing)."
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
        "--clip-at-goal",
        action="store_true",
        help=(
            "Trim each robot's printed path once it's parked at goal, keeping "
            "only the first arrival (output-only; doesn't affect solving/windowing). "
            "Useful for feeding paths to an external planner."
        ),
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
    run.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Base seed for --benchmark. Run k uses seed N+k, re-seeding the "
            "RNG and re-drawing the solver's random init (QAOA angles / neal "
            "seed) so runs differ but stay reproducible; each run's run_seed "
            "is saved. Omit for entropy-seeded runs (still logged)."
        ),
    )

    # ---- Visualization -----------------------------------------------------
    viz_g = parser.add_argument_group("Visualization (single-solve mode only)")
    viz_g.add_argument(
        "--visualize",
        nargs="?",
        const="animated",
        choices=["animated", "static", "steps"],
        default=None,
        metavar="MODE",
        help=(
            "Visualize the solved paths: 'animated' (default), 'static', or "
            "'steps'. Opens a browser window unless -o/--output is given."
        ),
    )
    viz_g.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="FILE",
        help=(
            "Save the visualization instead of opening a browser. Format from "
            "extension: .html (interactive), .gif (animated mode only), "
            ".png/.svg/.pdf (static image via kaleido)."
        ),
    )
    viz_g.add_argument(
        "--viz-discrete",
        action="store_true",
        help=(
            "Animate the raw discrete timeline (one frame per QUBO timestep) "
            "instead of smooth interpolated motion."
        ),
    )

    # ---- Config & misc -----------------------------------------------------
    misc = parser.add_argument_group("Config & misc")
    misc.add_argument(
        "--config",
        default=str(_HERE / "config/config.yaml"),
        metavar="FILE",
        help="Path to the main YAML config file (default: <package>/config/config.yaml)",
    )
    misc.add_argument(
        "--materials",
        default=str(_HERE / "config/materials.yaml"),
        metavar="FILE",
        help="Path to the materials YAML file (default: <package>/config/materials.yaml)",
    )
    misc.add_argument(
        "--verbose",
        "-v",
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


def resolve_h5_path(map_arg: str) -> str:
    """Normalize --map (with or without .h5/.yaml extension) to its .h5 path."""
    map_arg = str(map_arg)
    if map_arg.endswith(".h5") or map_arg.endswith(".yaml"):
        base_path = map_arg.rsplit(".", 1)[0]
    else:
        base_path = map_arg
    return f"{base_path}.h5"


def _parse_coord(token: str, context: str) -> tuple:
    """Parse an 'a,b' token into a (float, float) tuple. Left in whatever
    --coordinate-format's units; RobotConfig.resolve_coordinates() converts
    matrix/cartesian/world and casts to int once the problem's grid is known."""
    parts = token.split(",")
    if len(parts) != 2:
        raise ValueError(f"{context}: expected 'a,b', got '{token}'")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise ValueError(f"{context}: expected numeric 'a,b', got '{token}'")


def parse_robot_specs(raw: list[str], coordinate_format: str):
    """
    Parse --robot entries into a list of RobotConfig.
    Entry format: 'ID=START:GOAL[:PRIORITY[:SAFETY_RADIUS[:START_TIME]]]'
    """
    from quantum.robotConfiguration import RobotConfig

    robots = []
    for entry in raw:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --robot entry '{entry}'. Expected 'ID=START:GOAL[:PRIORITY[:SAFETY_RADIUS[:START_TIME]]]'."
            )
        robot_id, rest = entry.split("=", 1)
        robot_id = robot_id.strip()
        parts = rest.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid --robot entry '{entry}': needs at least START:GOAL.")

        start_tok, goal_tok, extra = parts[0], parts[1], parts[2:]
        if len(extra) > 3:
            raise ValueError(
                f"Invalid --robot entry '{entry}': too many fields "
                f"(expected START:GOAL[:PRIORITY[:SAFETY_RADIUS[:START_TIME]]])."
            )
        try:
            priority = float(extra[0]) if len(extra) > 0 else 1.0
            safety_radius = float(extra[1]) if len(extra) > 1 else 0.5
            start_time = int(extra[2]) if len(extra) > 2 else 0
        except ValueError:
            raise ValueError(f"Invalid --robot entry '{entry}': PRIORITY/SAFETY_RADIUS/START_TIME must be numeric.")

        start = _parse_coord(start_tok, f"--robot '{entry}' start")
        goal = _parse_coord(goal_tok, f"--robot '{entry}' goal")
        robots.append(RobotConfig(
            robot_id=robot_id, start=start, goal=goal,
            priority=priority, safety_radius=safety_radius, start_time=start_time,
            coordinate_format=coordinate_format,
        ))
    return robots


def parse_window_limits(raw: list[str], robot_ids) -> dict:
    """
    Parse window limit entries into {robot_id: max_steps}.

    Accepts either:
      - A single integer to cap all robots: ['6']
      - Per-robot pairs: ['robot_0=5', 'robot_1=3']
    """
    if not raw:
        return {}

    if len(raw) == 1 and "=" not in raw[0]:
        try:
            n = int(raw[0])
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid --window-limit value '{raw[0]}'. "
                f"Expected an integer or 'robot_id=N' pairs."
            )
        return {robot_id: n for robot_id in robot_ids}

    limits = {}
    for entry in raw:
        if "=" not in entry:
            raise argparse.ArgumentTypeError(
                f"Invalid --window-limit format '{entry}'. "
                f"Use a single integer for a global limit or 'robot_id=N' pairs."
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
        "K_hot": args.K_hot,
        "K_adj": args.K_adj,
        "K_start": args.K_start,
        "K_goal": args.K_goal,
        "K_lock": args.K_lock,
        "K_bt": args.K_bt,
        "K_tp": args.K_tp,
        "K_crash": args.K_crash,
        "K_swap": args.K_swap,
        "K_obs": args.K_obs,
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
        num_reads = (
            int(args.num_reads) if args.num_reads and args.num_reads != "auto" else 4
        )
        logger.minimal(f"Creating DWave solver (scale={norm_scale}, reads={num_reads})")
        return SolverFactory.create_solver(
            solver="dwave",
            normalize_scale=norm_scale,
            num_reads=num_reads,
        )

    elif args.solver == "pennylane":
        norm_scale = args.normalize_scale if args.normalize_scale is not None else 1.0
        num_reads = (
            int(args.num_reads)
            if args.num_reads and args.num_reads != "auto"
            else "auto"
        )

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
            machine=args.machine,
            threads=args.threads,
            use_session=not args.no_session,
            session_max_time=args.session_max_time,
        )

    elif args.solver == "qiskit_remote":
        # Mirrors the qiskit_hardware setup in qubo.py:
        #   SolverFactory.create_solver(solver="pennylane", device="qiskit.remote", ...)
        # normalize_scale defaults to 4.0 (same as qubo.py's qiskit_hardware).
        norm_scale = args.normalize_scale if args.normalize_scale is not None else 4.0
        num_reads = (
            int(args.num_reads)
            if args.num_reads and args.num_reads != "auto"
            else "auto"
        )
        device = args.device if args.device != "lightning.gpu" else "qiskit.remote"

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
            f"optimizer={args.optimizer}, steps={args.opt_steps}, scale={norm_scale}, "
            f"machine={args.machine or 'auto (least_busy)'})"
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
            machine=args.machine,
            use_session=not args.no_session,
            session_max_time=args.session_max_time,
        )

    elif args.solver == "qiskit_iqm":
        norm_scale = args.normalize_scale if args.normalize_scale is not None else 4.0
        num_reads = (
            int(args.num_reads)
            if args.num_reads and args.num_reads != "auto"
            else "auto"
        )
        device = args.device if args.device != "lightning.gpu" else "qiskit.iqm"

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
            f"Creating IQM solver via PennyLane "
            f"(device={device}, layers={args.layers}, "
            f"optimizer={args.optimizer}, steps={args.opt_steps}, scale={norm_scale}, "
            f"machine={args.machine or 'auto (smallest tier that fits)'})"
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
            machine=args.machine,
        )

    elif args.solver == "ilp":
        logger.minimal(
            f"Creating ILP solver (pyomo backend={args.pyomo_solver}, "
            f"time_limit={args.time_limit}s)"
        )
        return SolverFactory.create_solver(
            solver="ilp",
            pyomo_solver_name=args.pyomo_solver,
            time_limit=args.time_limit,
        )

    elif args.solver == "cbs":
        logger.minimal(
            f"Creating CBS solver (node_limit={args.node_limit}, "
            f"time_limit={args.time_limit}s)"
        )
        return SolverFactory.create_solver(
            solver="cbs",
            node_limit=args.node_limit,
            time_limit=args.time_limit,
        )

    else:
        raise ValueError(f"Unknown solver: {args.solver}")


def run_visualization(args: argparse.Namespace, problem, robot_paths: dict) -> None:
    """
    Render the solved paths per --visualize / --output.

    No --output: opens the figure in the default browser (plotly's fig.show()).
    With --output: saves to the file, format chosen by extension.
    """
    logger = get_logger()
    from quantum.visualizer import QuantumRoboticsVisualizer

    if not robot_paths:
        logger.minimal("[viz] No robot paths to visualize.")
        return

    mode = args.visualize
    out = args.output
    if out and out.lower().endswith(".gif") and mode != "animated":
        logger.minimal(
            f"[viz] GIF export requires the animated mode — switching from '{mode}'."
        )
        mode = "animated"

    viz = QuantumRoboticsVisualizer(
        (problem.grid.M, problem.grid.N),
        title=f"{args.problem} — {Path(args.map).name}",
    )
    obstacles = problem.grid.obstacles

    if mode == "static":
        fig = viz.create_static_plot(
            obstacles=obstacles, robot_paths=robot_paths, problem=problem
        )
    elif mode == "steps":
        fig = viz.create_step_by_step_plot(
            obstacles, robot_paths=robot_paths, problem=problem
        )
    else:  # animated
        fig = viz.create_animated_plot(
            obstacles=obstacles,
            robot_paths=robot_paths,
            problem=problem,
            smooth=not args.viz_discrete,
        )

    if not out:
        viz.show(fig)
    elif out.lower().endswith(".gif"):
        viz.write_gif(fig, out)
    elif out.lower().endswith((".html", ".htm")):
        viz.write_html(fig, out)
    else:
        viz.write_image(fig, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = build_parser()
    args = parser.parse_args()

    # -- Config --------------------------------------------------------------
    config = config_parser.load_config(
        args.config, sections=["penalty_sets", "verbose"]
    )

    # Verbose: CLI > config.yaml
    verbose_level = (
        args.verbose if args.verbose is not None else config["verbose"]["level"]
    )
    set_verbose_level(verbose_level)
    logger = get_logger()

    # -- Explicit robots (bypass the problems YAML) --------------------------
    if args.start is not None or args.goal is not None:
        if args.robot:
            parser.error("--start/--goal cannot be combined with --robot; use --robot robot_0=START:GOAL instead.")
        if args.start is None or args.goal is None:
            parser.error("--start and --goal must be given together.")
        args.robot = [f"robot_0={args.start}:{args.goal}"]

    logger.minimal(f"qubo_cli starting | map={args.map} | problem={args.problem}")

    # -- Materials -----------------------------------------------------------
    materials_data = config_parser.load_config(args.materials)["materials"]

    # -- Problem -------------------------------------------------------------
    if args.robot:
        h5_path = resolve_h5_path(args.map)
        try:
            robots = parse_robot_specs(args.robot, args.coordinate_format)
        except (ValueError, OSError, KeyError) as e:
            logger.minimal(f"[ERROR] {e}")
            sys.exit(1)
        logger.minimal(
            f"Using {len(robots)} explicitly-specified robot(s) from {h5_path} -- --problem ignored"
        )
        try:
            problem = PathfindingProblem.from_h5(
                h5_path,
                robots=robots,
                materials_data=materials_data,
                T=args.horizon,
            )
        except InfeasibleProblemError as e:
            logger.minimal(f"[ERROR] {e}")
            sys.exit(1)
    else:
        try:
            problem = PathfindingProblem.from_map_config(
                args.map,
                problem_name=args.problem,
                materials_data=materials_data,
                coordinate_format=args.coordinate_format,
            )
        except InfeasibleProblemError as e:
            logger.minimal(f"[ERROR] {e}")
            sys.exit(1)

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
    window_limits = parse_window_limits(args.window_limit, problem.robots.keys())

    # -- Builder -------------------------------------------------------------
    builder_kwargs = {
        "penalties": penalties,
        "name": args.problem,
        "robot_window_limits": window_limits if window_limits else None,
        "log_reductions": not args.no_reduction_log,
    }
    if args.var_limit is not None:
        builder_kwargs["var_limit"] = args.var_limit

    if args.solver == "ilp":
        # ILP has no penalty weights, var_limit, or windowing — builder_kwargs
        # (penalties/var_limit/robot_window_limits/log_reductions) don't apply.
        if args.builder == "grid":
            p = problem.as_grid_only()
            builder = GridILPBuilder(p, name=args.problem, verbose_level=verbose_level)
        else:  # graph
            p = problem.as_graph_only()
            builder = GraphILPBuilder(p, name=args.problem, verbose_level=verbose_level)
    elif args.solver == "cbs":
        # CBS has no penalty weights, var_limit, or windowing either — same
        # reasoning as ILP above.
        if args.builder == "grid":
            p = problem.as_grid_only()
            builder = GridCBSBuilder(p, name=args.problem, verbose_level=verbose_level)
        else:  # graph
            p = problem.as_graph_only()
            builder = GraphCBSBuilder(p, name=args.problem, verbose_level=verbose_level)
    elif args.builder == "grid":
        p = problem.as_grid_only()
        builder_kwargs["distance_scaling"] = args.distance_scaling
        builder = QUBOBuilder(p, **builder_kwargs)
    else:  # graph
        p = problem.as_graph_only()
        builder = GraphQUBO(p, **builder_kwargs)

    logger.minimal(
        f"Builder: {args.builder.upper()} | window_limits={window_limits or 'none'}"
    )

    # -- Solver --------------------------------------------------------------
    solver = build_solver(args, verbose_level)

    # -- Run mode ------------------------------------------------------------
    if args.benchmark:
        if args.visualize:
            logger.minimal(
                "[viz] --visualize is only available in single-solve mode; ignoring."
            )
        logger.minimal(
            f"Running benchmark: {args.num_runs} runs, level {args.benchmark_level}"
        )
        runner = bm_module.BenchmarkRunner(
            builder,
            solver,
            num_runs=args.num_runs,
            level=args.benchmark_level,
            preprocess=_resolve_preprocess(args),
            seed=args.seed,
        )
        runner.run_build()

    else:
        # Single solve
        timer = time.time()
        # ILP builders rebuild themselves inside solver.solve() (see
        # ILPSolver.solve()) so the preprocess flag always takes effect;
        # pre-building here would just duplicate work and logging. Same for
        # QUBO builders under preprocess=True: _prepare_window() builds after
        # BFS reduction populates _active_cells, so a pre-build here would be
        # a full-grid build immediately discarded. Only preprocess=False
        # needs it, since that path reads builder.Q directly.
        if not hasattr(builder, "local_index") and not (
            preprocess_modes.uses_windowed_pipeline(_resolve_preprocess(args))
        ):
            builder.build()
        solution = solver.solve(builder, preprocess=_resolve_preprocess(args))
        # Use p (the grid-only/graph-only problem actually passed to the
        # builder), not problem
        path = solver.decode_path(solution["solution"], p)

        energy = solution["energy"]
        if isinstance(energy, list):
            energy = sum(energy)

        logger.debug(
            f"Raw path:  {path}"
        )  # Full decoded tuples — only useful at verbose=3
        logger.minimal(f"Energy:    {energy:.4f}")
        logger.minimal(f"Time:      {time.time() - timer:.4f}")

        for robot_id, robot in problem.robots.items():
            robot_path = robot.path
            if args.clip_at_goal:
                robot_path = clip_path_at_goal(robot_path, tuple(robot.goal))
            formatted_path = [
                (*robot.format_position((i, j)), t) for i, j, t in robot_path
            ]
            logger.minimal(f"  [{robot_id}] {formatted_path}")

        if args.visualize:
            # visualizer.py expects native matrix (row, col) input regardless of
            # --coordinate-format, so pass the raw decoded path, not a formatted one.
            run_visualization(args, problem, solver.get_robot_paths(path))


if __name__ == "__main__":
    main()
