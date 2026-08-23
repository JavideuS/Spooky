from pathlib import Path
from datetime import datetime
import heapq
import json
import time
from quantum.utils.validation import is_valid_move, get_position_representation
from quantum.utils.logger import get_logger


class BenchmarkRunner:
    def __init__(
        self,
        qubobuilder,
        solver,
        num_runs=10,
        output_dir="results/benchmarks",
        level=2,
        preprocess=True,
    ):
        """
        Run benchmark on a given solver and problem.

        Args:
            qubobuilder (QUBOBuilder): A fully initialized QUBO builder
            solver (QUBOSolver): A solver implementing `.solve(Q)`
            num_runs (int): Number of times to run the solver
            output_dir (str): Where to save results
            level (int): Benchmark verbosity level (1=Summary, 2=Paths, 3=Full)
                - Level 1: Only statistics, timing, energy, validation pass/fail
                - Level 2: Level 1 + robot paths and per-robot validation details
                - Level 3: Level 2 + raw bit solution for debugging
            preprocess (bool): Forwarded to solver.solve() each run — toggles
                each solver's own variable-reduction step (BFS reachability
                pruning for both QUBO and ILP, plus diagonal fixing/windowing
                for QUBO specifically).
        """
        self.builder = qubobuilder
        self.problem = qubobuilder.problem
        self.penalty_set = qubobuilder.penalties
        self.solver = solver
        self.num_runs = num_runs
        self.preprocess = preprocess
        self.level = max(1, min(3, level))  # Clamp to 1-3
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()  # Use global logger level

        # Store metadata once, not per-run
        self.results = {
            "metadata": {
                "problem": self.problem.to_dict(),
                "solver": self.solver.to_dict(),
                "penalty_set": self.penalty_set,
                "benchmark_level": self.level,
                "num_runs": num_runs,
                "timestamp": datetime.now().isoformat(),
            },
            "runs": [],
        }

    def run_build(self):
        """Run the benchmark multiple times and store results"""
        self.logger.minimal(f"\nBenchmarking Problem: {self.problem.name}")
        self.logger.minimal(f"Using Solver: {self.solver.name}")
        self.logger.minimal(f"Penalty Set: {self.penalty_set.get('name', 'unnamed')}")
        self.logger.minimal(
            f"Benchmark Level: {self.level} ({'Summary' if self.level == 1 else 'Paths' if self.level == 2 else 'Full'})"
        )
        self.logger.minimal("-" * 60)

        valid_count = 0
        total_solve_time = 0.0
        total_estimated_qpu_time_gate = 0.0
        runs_with_gate_estimate = 0
        total_estimated_qpu_time_clops = 0.0
        runs_with_clops_estimate = 0
        total_billed_usage_sec = 0.0
        runs_with_billed_usage = 0
        total_queue_time_sec = 0.0
        runs_with_queue_time = 0
        total_overhead_sec = 0.0

        # Run multiple trials
        for run_id in range(1, self.num_runs + 1):
            self.builder.reset_problem()
            build_start = time.time()
            # ILP builders rebuild themselves inside solver.solve() so the
            # preprocess flag always takes effect (see ILPSolver.solve());
            # pre-building here would just be redundant work and duplicate
            # logging. QUBO builders still need this: their preprocess=False
            # path reads builder.Q directly without calling build() itself.
            if not hasattr(self.builder, "local_index"):
                self.builder.build()
            build_duration = time.time() - build_start

            solve_start = time.time()
            solution = self.solver.solve(self.builder, preprocess=self.preprocess)
            solve_duration = time.time() - solve_start

            self.logger.minimal(
                f"Build time: {build_duration:.4f}s, Solve time: {solve_duration:.4f}s"
            )

            # Decode path for validation
            path = self.solver.decode_path(solution["solution"], self.problem)
            validation = is_solution_valid(path, self.problem)

            # If invalid, work out whether pre-processing forced the conflict
            # (bypassing K_crash/K_swap entirely) or the solver actually
            # sampled a bitstring that violates a penalty that was present.
            forced_collisions = solution.get("metadata", {}).get(
                "forced_collisions", []
            )
            invalid_cause = _attribute_invalid_cause(validation, forced_collisions)
            if invalid_cause:
                validation["invalid_cause"] = invalid_cause

            # Calculate total energy (handle both scalar and list energies)
            if isinstance(solution["energy"], list):
                total_energy = sum(solution["energy"])
            else:
                total_energy = solution["energy"]

            valid_count += int(validation["valid"])
            total_solve_time += solve_duration

            # Build result based on level
            result = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "valid": validation["valid"],
                "energy": total_energy,
                "build_time_sec": round(build_duration, 3),
                "execution_time_sec": round(solve_duration, 3),
            }
            if invalid_cause:
                result["invalid_cause"] = invalid_cause

            # Exact solvers (ILP, CBS) report whether they actually proved
            # optimality or just returned their best satisfy solution on a
            # time/node-limit timeout
            termination_condition = solution.get("metadata", {}).get(
                "termination_condition"
            )
            if termination_condition is not None:
                result["termination_condition"] = termination_condition

            # Pre-execution QPU time estimates + real hardware telemetry. Each
            # window entry has "device" ("qiskit.remote"|"qiskit.iqm") plus
            # "wall_clock_sec" (time.time() around the hardware call) and
            # either IBM's {"gate_model", "clops_model", "billed_usage"} or
            # IQM's {"iqm_timing": {...}} — see quantum.hardware.README.md.
            qpu_time_estimates = solution.get("metadata", {}).get(
                "qpu_time_estimates", []
            )
            gate_estimates = [
                e["gate_model"] for e in qpu_time_estimates if e and e.get("gate_model")
            ]
            clops_estimates = [
                e["clops_model"]
                for e in qpu_time_estimates
                if e and e.get("clops_model")
            ]
            if gate_estimates:
                result["estimated_qpu_time_gate_model_sec"] = round(
                    sum(e["total_estimate_sec"] for e in gate_estimates), 4
                )
            if clops_estimates:
                result["estimated_qpu_time_clops_model_sec"] = round(
                    sum(e["total_estimate_sec"] for e in clops_estimates), 4
                )

            billed_usage_values = [
                e["billed_usage"]["usage"]
                for e in qpu_time_estimates
                if e and e.get("billed_usage") and e["billed_usage"].get("usage") is not None
            ]
            if billed_usage_values:
                result["billed_usage_sec"] = sum(billed_usage_values)

            # Build/execution/queue/overhead split — see
            # _compute_hardware_time_split's docstring for why
            # execution_time_sec is the real-execution reference directly,
            # not solve_duration minus queue_time_sec (that silently folds
            # backend/session setup and classical preprocessing into
            # "execution"). None (and no split applied) if no window has
            # enough data — e.g. a classical/simulator solve.
            hw_execution_sec, queue_time_sec = _compute_hardware_time_split(
                qpu_time_estimates
            )
            if queue_time_sec is not None:
                result["execution_time_sec"] = round(hw_execution_sec, 3)
                result["queue_time_sec"] = round(queue_time_sec, 3)
                result["overhead_sec"] = round(
                    max(0.0, solve_duration - hw_execution_sec - queue_time_sec), 3
                )

            if self.level >= 2 and qpu_time_estimates:
                result["qpu_time_estimate_breakdown"] = qpu_time_estimates

            # Add variable stats from solver based on level
            window_stats = solution.get("metadata", {}).get("window_stats", [])
            if window_stats:
                # Always compute totals
                total_initial = sum(ws["initial_variables"] for ws in window_stats)
                total_reduced = sum(ws["variables_reduced"] for ws in window_stats)
                total_final = sum(ws["final_variables"] for ws in window_stats)
                avg_reduction = (
                    total_reduced / total_initial if total_initial > 0 else 0
                )

                # Level 1: Just totals (aggregated across all windows)
                result["variable_stats"] = {
                    "total_initial_variables": total_initial,
                    "total_variables_reduced": total_reduced,
                    "total_final_variables": total_final,
                    "average_reduction_ratio": round(avg_reduction, 4),
                    "num_windows": len(window_stats),
                }

                # Level 2+: Add detailed per-window breakdown
                if self.level >= 2:
                    result["window_variable_stats"] = window_stats

            # Level 2+: Add robot paths and validation details
            if self.level >= 2:
                # Extract robot paths from robot.path (already corrected/merged)
                robot_paths = {}
                for robot_id, robot in self.problem.robots.items():
                    robot_paths[robot_id] = robot.path

                result["robot_paths"] = robot_paths

                # Add per-robot validation details
                validation_details = {}
                for key, value in validation.get("details", {}).items():
                    if key.startswith("robot_"):
                        validation_details[key] = value

                result["validation_details"] = validation_details

                result["solution_statistics"] = _compute_solution_statistics(
                    self.problem, robot_paths, validation_details, validation["valid"]
                )

                # Also store per-window energies for analysis
                if isinstance(solution["energy"], list):
                    result["window_energies"] = solution["energy"]

            # Level 3: Add raw bit solution for debugging
            if self.level >= 3:
                result["raw_solution"] = solution["solution"]

            self.results["runs"].append(result)

            if "estimated_qpu_time_gate_model_sec" in result:
                total_estimated_qpu_time_gate += result[
                    "estimated_qpu_time_gate_model_sec"
                ]
                runs_with_gate_estimate += 1
            if "estimated_qpu_time_clops_model_sec" in result:
                total_estimated_qpu_time_clops += result[
                    "estimated_qpu_time_clops_model_sec"
                ]
                runs_with_clops_estimate += 1
            if "billed_usage_sec" in result:
                total_billed_usage_sec += result["billed_usage_sec"]
                runs_with_billed_usage += 1
            if "queue_time_sec" in result:
                total_queue_time_sec += result["queue_time_sec"]
                total_overhead_sec += result["overhead_sec"]
                runs_with_queue_time += 1

            status = "✅ Valid" if validation["valid"] else "❌ Invalid"
            if not validation["valid"]:
                self.logger.standard(
                    "Validation details:", validation.get("details", {})
                )
                self.logger.standard("Reason:", validation.get("reason", "unknown"))
                self.logger.standard("Message:", validation.get("message", ""))
                if invalid_cause:
                    self.logger.standard("Cause:", invalid_cause)
            qpu_estimate_parts = []
            if "estimated_qpu_time_gate_model_sec" in result:
                qpu_estimate_parts.append(
                    f"gate={result['estimated_qpu_time_gate_model_sec']:.2f}s"
                )
            if "estimated_qpu_time_clops_model_sec" in result:
                qpu_estimate_parts.append(
                    f"clops={result['estimated_qpu_time_clops_model_sec']:.2f}s"
                )
            if "billed_usage_sec" in result:
                qpu_estimate_parts.append(f"billed={result['billed_usage_sec']:.2f}s")
            qpu_estimate_str = (
                f" | Est. QPU time ({', '.join(qpu_estimate_parts)})"
                if qpu_estimate_parts
                else ""
            )
            # build/execution/queue/overhead only splits apart once
            # queue_time_sec is computable (hardware runs with enough
            # per-window data) — a classical/simulator run just shows a
            # plain solve time.
            if "queue_time_sec" in result:
                time_str = (
                    f"build={result['build_time_sec']:.2f}s, "
                    f"exec={result['execution_time_sec']:.2f}s, "
                    f"queue={result['queue_time_sec']:.2f}s, "
                    f"overhead={result['overhead_sec']:.2f}s"
                )
            else:
                time_str = f"{solve_duration:.2f}s"
            self.logger.minimal(
                f"Run {run_id}: {status} | Time: {time_str}"
                f"{qpu_estimate_str} | Energy: {total_energy:.4f}"
            )
            self.logger.minimal(f"Path: {path}")
            for robot_id, robot in self.problem.robots.items():
                self.logger.minimal(f" Robot {robot_id} path: {robot.path}")

        avg_solve_time = total_solve_time / self.num_runs if self.num_runs else 0
        self.logger.minimal(
            f"\nAccuracy: {valid_count}/{self.num_runs} valid "
            f"({valid_count / self.num_runs:.1%}) | "
            f"Average solve time: {avg_solve_time:.3f}s"
        )
        self.results["summary"] = {
            "valid_runs": valid_count,
            "total_runs": self.num_runs,
            "accuracy": round(valid_count / self.num_runs, 4) if self.num_runs else 0,
            "average_solve_time_sec": round(avg_solve_time, 4),
        }
        if runs_with_gate_estimate:
            self.results["summary"]["average_estimated_qpu_time_gate_model_sec"] = (
                round(total_estimated_qpu_time_gate / runs_with_gate_estimate, 4)
            )
        if runs_with_clops_estimate:
            self.results["summary"]["average_estimated_qpu_time_clops_model_sec"] = (
                round(total_estimated_qpu_time_clops / runs_with_clops_estimate, 4)
            )
        if runs_with_billed_usage:
            self.results["summary"]["total_billed_qpu_usage_sec"] = round(
                total_billed_usage_sec, 4
            )
            self.logger.minimal(
                f"Total billed QPU usage this benchmark: {total_billed_usage_sec:.2f}s "
                f"(across {runs_with_billed_usage}/{self.num_runs} run(s) with usage data)"
            )
        if runs_with_queue_time:
            self.results["summary"]["total_queue_time_sec"] = round(
                total_queue_time_sec, 4
            )
            self.results["summary"]["total_overhead_sec"] = round(
                total_overhead_sec, 4
            )

        self.save_results()
        return self.results

    def save_results(self):
        filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_results = convert_tuple_keys_to_str(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        self.logger.minimal(f"\nBenchmark complete. Results saved to {filepath}")


def _compute_hardware_time_split(qpu_time_estimates):
    """
    Split real-hardware wall-clock time into (execution, queue), summed
    across all windows in one run.

    For each window, the best available *real* execution-time reference —
    never a raw estimate when ground truth exists:
        1. IBM: job.usage() (billed_usage) — actual billed QPU time.
        2. IQM: iqm_timing.job_total_sec — the job's own server-processing
           span (received -> ready), covering compile+validation+execution.
        3. Fallback: the gate-model pre-execution estimate, only if neither
           of the above is available for that window.
    execution_time_sec is the sum of that reference directly. queue_time_sec
    is wall-clock hardware-call time (Pennylane_solver.py's sample_time,
    i.e. time.time() around the sampling call) minus that same reference.

    Deliberately NOT `solve_duration - queue_time_sec`: solve_duration also
    includes backend/session setup and classical per-window preprocessing,
    neither of which is queue time OR real QPU execution — subtracting only
    queue time from it would silently misattribute that overhead as
    "execution" (exactly what this function replaces; see the "IBM quota"
    section of quantum/hardware/README.md for the run that exposed it: 2s
    billed usage vs. an "execution_time_sec" of 11s from the old formula).
    Callers should treat solve_duration - execution_time_sec - queue_time_sec
    as its own overhead bucket rather than folding it into either of these.

    A window contributes nothing if it has no wall_clock_sec at all (e.g. a
    fully pre-processed window that never touched hardware).

    Returns:
        (float, float) | (None, None): (execution_time_sec, queue_time_sec)
        in seconds, or (None, None) if no window in this run had enough data
        to compute either (e.g. a classical/simulator solve, where neither
        concept applies at all).
    """
    execution_total = 0.0
    queue_total = 0.0
    counted_any = False
    for entry in qpu_time_estimates:
        if not entry:
            continue
        wall_clock_sec = entry.get("wall_clock_sec")
        if wall_clock_sec is None:
            continue

        real_reference_sec = None
        billed_usage = entry.get("billed_usage")
        if billed_usage and billed_usage.get("usage") is not None:
            real_reference_sec = billed_usage["usage"]
        elif entry.get("iqm_timing") and entry["iqm_timing"].get("job_total_sec") is not None:
            real_reference_sec = entry["iqm_timing"]["job_total_sec"]
        elif entry.get("gate_model"):
            real_reference_sec = entry["gate_model"]["total_estimate_sec"]

        if real_reference_sec is None:
            continue

        execution_total += real_reference_sec
        queue_total += max(0.0, wall_clock_sec - real_reference_sec)
        counted_any = True

    return (execution_total, queue_total) if counted_any else (None, None)


_FIX_MECHANISM_LABELS = {
    "bfs": "bfs_fixing",
    "diag": "diag_fixing",
    "locked_inactive": "goal_lock",
}


def _graph_shortest_distance(graph, start_node, goal_node):
    """Dijkstra shortest-path distance between two node indices over a
    quantum.map.Graph's adjacency (node -> set of (neighbor, weight)).

    Used as the "optimal length" baseline for graph-problem path
    efficiency — abs(goal_node - start_node) only means anything if node
    IDs happen to be laid out linearly along the path, which isn't true
    for a real graph topology."""
    if start_node == goal_node:
        return 0.0
    best = {start_node: 0.0}
    visited = set()
    heap = [(0.0, start_node)]
    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == goal_node:
            return dist
        for neighbor, weight in graph.adjacency.get(node, ()):
            new_dist = dist + weight
            if new_dist < best.get(neighbor, float("inf")):
                best[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    return float("inf")  # goal unreachable from start


def _compute_solution_statistics(
    problem, robot_paths, validation_details, overall_valid
):
    """
    Per-robot and aggregate path-quality statistics for one run: path
    length, whether the goal was reached, and path efficiency (optimal
    length / moves actually taken). Complements is_solution_valid()'s
    pass/fail check with a continuous quality measure.

    robot_paths: {robot_id: [(i, j, t), ...]} as stored on
        problem.robots[...].path. path[0] is always the robot's start
        position (decode_path() includes it), so "moves taken" is
        len(path) - 1, not len(path) — dividing optimal_length (a move
        count) by len(path) directly undercounts efficiency by one step
        for every real path.
    validation_details: is_solution_valid()'s per-robot {"robot_N": {...}}
        entries. Only present for robots is_solution_valid actually
        reached (it returns early on the first individually-invalid
        robot) — overall_valid is the fallback for any robot it didn't
        get to.
    """
    robot_ids = list(problem.robots.keys())
    stats = {"total_robots": problem.num_robots, "robot_statistics": {}}

    successful = 0
    for robot_num, robot_id in enumerate(robot_ids):
        robot = problem.robots[robot_id]
        path = robot_paths.get(robot_id, [])
        detail = validation_details.get(f"robot_{robot_num}")
        robot_valid = detail["valid"] if detail is not None else overall_valid
        successful += int(robot_valid)

        moves_taken = max(0, len(path) - 1)
        robot_stats = {
            "path_length": len(path),
            "moves_taken": moves_taken,
            "goal_reached": robot.is_at_goal(),
            "validation_passed": robot_valid,
            "priority": robot.priority,
            "start_time": robot.start_time,
            "time_horizon": robot.T,
        }

        if problem.grid is not None:
            optimal_length = problem.manhattan_distance(robot.start, robot.goal)
        elif isinstance(robot.start, int) and isinstance(robot.goal, int):
            optimal_length = _graph_shortest_distance(
                problem.graph, robot.start, robot.goal
            )
        else:
            optimal_length = problem.euclidean_distance(robot.start, robot.goal)
        robot_stats["optimal_path_length"] = optimal_length

        if optimal_length == 0:
            robot_stats["path_efficiency"] = 1.0  # already at goal
        elif moves_taken > 0:
            robot_stats["path_efficiency"] = optimal_length / moves_taken
        else:
            robot_stats["path_efficiency"] = 0.0  # never moved but should have

        stats["robot_statistics"][robot_id] = robot_stats

    stats["successful_robots"] = successful
    stats["success_rate"] = (
        successful / problem.num_robots if problem.num_robots else 0.0
    )

    if problem.num_robots > 1:
        lengths = [s["path_length"] for s in stats["robot_statistics"].values()]
        efficiencies = [
            s["path_efficiency"] for s in stats["robot_statistics"].values()
        ]
        stats["aggregate"] = {
            "total_path_length": sum(lengths),
            "avg_path_length": sum(lengths) / len(lengths),
            "max_path_length": max(lengths),
            "min_path_length": min(lengths),
            "avg_efficiency": sum(efficiencies) / len(efficiencies),
        }

    return stats


def _attribute_invalid_cause(validation, forced_collisions):
    """
    Cross-reference a failed validation's reported vertex conflicts against
    the solver's pre-processing forced-collision log (BaseSolver._flag_forced_collisions,
    surfaced via solution["metadata"]["forced_collisions"]).

    A run can be invalid for two very different reasons that call for
    different fixes: the conflicting cell/time was already forced before
    K_crash/K_swap ever ran (a pre-processing routing gap), or the QAOA/
    annealing sample itself landed on a degenerate state that violates a
    penalty which was genuinely present in Q (a solver-convergence issue).

    Returns None if validation passed, otherwise a dict:
    {"origin": "solver_sampling"} or
    {"origin": "pre_processing", "matches": [{"cell", "time", "robots", "fixed_by"}, ...]}
    where "fixed_by" names the mechanism(s) that forced the conflict — e.g.
    "diag_fixing" when both robots were forced by the same mechanism, or
    "diag_fixing + goal_lock" when one robot was still being fixed while the
    other was an already-finished robot locked at its goal.
    """
    if validation.get("valid", True):
        return None

    forced_by_cell_time = {}
    for fc in forced_collisions:
        forced_by_cell_time.setdefault((fc["cell"], fc["time"]), []).append(fc)

    matched = []
    for conflict in validation.get("details", {}).get("conflicts", []):
        key = (conflict["cell"], conflict["time"])
        if key in forced_by_cell_time:
            matched.extend(forced_by_cell_time[key])

    if not matched:
        # swap_conflicts have no pre-processing equivalent today (forced_collisions
        # only ever records same-cell/same-time fixes), so any swap_conflict here
        # is necessarily solver-side.
        return {"origin": "solver_sampling"}

    matches = []
    for fc in matched:
        labels = {
            _FIX_MECHANISM_LABELS.get(source, source) for _, source in fc["sources"]
        }
        matches.append(
            {
                "cell": fc["cell"],
                "time": fc["time"],
                "robots": fc["robots"],
                "fixed_by": " + ".join(sorted(labels)),
            }
        )

    return {"origin": "pre_processing", "matches": matches}


def is_solution_valid(solution, problem):
    """
    Checks if the decoded path represents a valid path from start to goal.
    Accepts a list of ((i, j, t), robot_num) tuples (decoded path), or a list of such paths.
    Args:
        solution (list): List of ((i, j, t), robot_num) tuples,
            or list of such lists.
        problem: Problem instance (grid or graph)
    Returns:
        dict: Validation result with 'valid' flag and optional error details
    """
    problem_type = problem.get_format_type()

    if problem_type != "grid" and problem_type != "graph":
        return {
            "valid": False,
            "reason": "unsupported_problem_type",
            "message": f"Unsupported problem type: {problem_type}",
        }

    result = {"valid": True, "details": {}}

    # If input is a list of paths (list of lists of tuples)
    if solution and isinstance(solution[0], list):
        return [is_solution_valid(path, problem) for path in solution]

    # If input is a single path (list of tuples)
    positions = list(solution)
    if not positions:
        result["valid"] = False
        result["reason"] = "empty_path"
        result["message"] = "❌ No path positions found. Invalid sample"
        return result

    result["details"]["path"] = positions

    # Group positions by robot
    robot_positions = {}
    for (i, j, t), robot_num in positions:
        if robot_num not in robot_positions:
            robot_positions[robot_num] = []
        robot_positions[robot_num].append((i, j, t))

    # Validate each robot's path using unified validation
    for robot_num, robot_path in robot_positions.items():
        robot_id = list(problem.robots.keys())[robot_num]
        robot_result = _validate_single_robot_path_unified(
            robot_path, problem, robot_id, robot_num
        )
        if not robot_result["valid"]:
            result["valid"] = False
            result["reason"] = f"robot_{robot_num}_invalid"
            result["message"] = f"❌ Robot {robot_num}: {robot_result['message']}"
            result["details"][f"robot_{robot_num}"] = robot_result
            return result
        result["details"][f"robot_{robot_num}"] = robot_result

    # multi-robot vertex conflict check (same cell at same time)
    occupancy = {}  # (i, j, t) -> [robot_num, ...]
    for robot_num, robot_path in robot_positions.items():
        for i, j, t in robot_path:
            key = (i, j, t)
            occupancy.setdefault(key, []).append(robot_num)

    # Find collisions where two or more robots occupy same cell at same time
    conflicts = []
    for (i, j, t), robots in occupancy.items():
        if len(robots) > 1:
            conflicts.append({"cell": (i, j), "time": t, "robots": sorted(robots)})

    # multi-robot swap (edge) conflict check: two robots exchange cells
    # between consecutive timesteps (robot A: X@t -> Y@t+1, robot B: Y@t -> X@t+1).
    # This is invisible to the vertex check above since neither robot ever
    # shares a cell at the same time — they cross mid-edge instead.
    position_by_time = {
        robot_num: {t: (i, j) for (i, j, t) in robot_path}
        for robot_num, robot_path in robot_positions.items()
    }
    robot_nums_sorted = sorted(position_by_time.keys())
    swap_conflicts = []
    for idx, r1 in enumerate(robot_nums_sorted):
        for r2 in robot_nums_sorted[idx + 1 :]:
            times_r1 = position_by_time[r1]
            times_r2 = position_by_time[r2]
            for t in set(times_r1) & set(times_r2):
                if (t + 1) not in times_r1 or (t + 1) not in times_r2:
                    continue
                if times_r1[t] == times_r2[t + 1] and times_r2[t] == times_r1[t + 1]:
                    swap_conflicts.append(
                        {
                            "cells": (times_r1[t], times_r2[t]),
                            "time": (t, t + 1),
                            "robots": [r1, r2],
                        }
                    )

    if conflicts or swap_conflicts:
        conflicts.sort(key=lambda c: (c["time"], c["cell"]))
        swap_conflicts.sort(key=lambda c: (c["time"], c["robots"]))
        result["valid"] = False
        reasons = []
        messages = []
        if conflicts:
            reasons.append("vertex_conflict")
            messages.append(f"{len(conflicts)} vertex collision(s)")
            result["details"]["conflicts"] = conflicts
        if swap_conflicts:
            reasons.append("swap_conflict")
            messages.append(f"{len(swap_conflicts)} swap collision(s)")
            result["details"]["swap_conflicts"] = swap_conflicts
        result["reason"] = "+".join(reasons)
        result["message"] = f"❌ Multi-robot conflict detected: {', '.join(messages)}"
        return result

    result["valid"] = True
    result["message"] = "✅ Solution is valid"
    return result


def _validate_single_robot_path_unified(positions, problem, robot_id, robot_num):
    """
    Unified validation function for single robot paths in both grid and graph
    problems.

    Args:
        positions: List of (i, j, t) tuples for one robot
        problem: Problem instance (grid or graph)
        robot_id: Robot identifier/name
        robot_num: Robot number for error messages

    Returns:
        dict: Validation result
    """
    result = {"valid": True, "details": {}}

    # Get notation abstraction based on problem type
    notation = _get_notation_abstraction(problem)

    # Get robot's individual timeline end
    robot = problem.robots[robot_id]
    robot_end_time = robot.T + robot.start_time

    if not positions:
        result["valid"] = False
        result["reason"] = "empty_path"
        result["message"] = f"❌ Robot {robot_num}: No path positions found"
        return result

    # 1. Sort by time step
    positions.sort(key=lambda x: x[2])

    # 2. Get expected time range and check all time steps are present
    expected_times = notation.get_expected_time_range(problem, robot_id, robot_end_time)
    expected_goal = notation.get_goal_position(problem, robot_id)

    # Early stop support: Check if robot reached goal early
    goal_times = [
        t
        for i, j, t in positions
        if get_position_representation(problem, (i, j)) == expected_goal
    ]

    if goal_times:
        first_goal_time = min(goal_times)
        # Only require timesteps up to when goal was first reached
        expected_times = set(t for t in expected_times if t <= first_goal_time)

        # Verify robot stayed at goal if it has positions after reaching it
        later_positions = [(i, j, t) for i, j, t in positions if t > first_goal_time]
        for i, j, t in later_positions:
            if get_position_representation(problem, (i, j)) != expected_goal:
                result["valid"] = False
                result["reason"] = "left_goal_after_reaching"
                result["message"] = (
                    f"❌ Robot {robot_num}: Left goal position at time {t} "
                    f"after reaching it at time {first_goal_time}"
                )
                result["details"]["first_goal_time"] = first_goal_time
                result["details"]["left_at_time"] = t
                return result

    all_times = set(t for _, _, t in positions)
    missing_times = expected_times - all_times
    # Only flag as extra if timesteps exceed the robot's full timeline range
    # Robot can have timesteps from start_time to (start_time + T - 1)
    # Extra timesteps are those >= (start_time + T)
    extra_times = set(t for t in all_times if t >= robot_end_time)

    if missing_times:
        result["valid"] = False
        result["reason"] = "missing_time_steps"
        result["message"] = f"❌ Robot {robot_num}: Missing time steps: {missing_times}"
        result["details"]["missing_times"] = list(missing_times)
        return result

    if extra_times:
        result["valid"] = False
        result["reason"] = "extra_time_steps"
        result["message"] = (
            f"❌ Robot {robot_num}: Timesteps exceed robot's timeline "
            f"[{robot.start_time}, {robot_end_time - 1}]: {extra_times}"
        )
        result["details"]["extra_times"] = list(extra_times)
        result["details"]["robot_timeline"] = (
            f"[{robot.start_time}, {robot_end_time - 1}]"
        )
        return result

    # 3. One-hot constraint per time step
    time_to_positions = {}
    for i, j, t in positions:
        time_to_positions.setdefault(t, []).append((i, j))

    for t, cells in time_to_positions.items():
        if len(cells) > 1:
            result["valid"] = False
            result["reason"] = "multiple_positions_per_time"
            result["message"] = (
                f"❌ Robot {robot_num}: Multiple positions at time {t}: {cells}"
            )
            result["details"]["conflicts"] = {t: cells}
            return result

    # 4. Start position check
    start_time = notation.get_start_time(problem, robot_id)
    start_positions = [(i, j) for i, j, t in positions if t == start_time]
    if not start_positions:
        result["valid"] = False
        result["reason"] = "wrong_start"
        result["message"] = (
            f"❌ Robot {robot_num}: No start position found at time {start_time}"
        )
        return result

    expected_start = notation.get_start_position(problem, robot_id)
    actual_start = get_position_representation(problem, start_positions[0])
    if actual_start != expected_start:
        result["valid"] = False
        result["reason"] = "wrong_start"
        result["message"] = (
            f"❌ Robot {robot_num}: Wrong start position. Expected {expected_start}, got {actual_start}"
        )
        result["details"]["expected_start"] = expected_start
        result["details"]["actual_start"] = actual_start
        return result

    # 5. Goal position check (already verified in early stop logic above)
    # If goal_times is empty, goal was never reached
    if not goal_times:
        result["valid"] = False
        result["reason"] = "goal_not_reached"
        result["message"] = (
            f"❌ Robot {robot_num}: Goal position {expected_goal} never reached"
        )
        result["details"]["goal"] = expected_goal
        return result

    # 6. Movement must be valid (adjacent positions only)
    last_pos = None
    for i, j, t in positions:
        current_pos = (i, j)

        # Check for obstacles (grid only)
        if notation.has_obstacle_check():
            obstacles = notation.get_obstacles(problem)
            if current_pos in obstacles:
                result["valid"] = False
                result["reason"] = "obstacle_collision"
                result["message"] = (
                    f"❌ Robot {robot_num}: Path goes through obstacle at {current_pos} at time {t}"
                )
                result["details"]["collisions"] = [(*current_pos, t)]
                return result

        if last_pos is not None:
            last_i, last_j, last_t = last_pos

            # Goal-lock bypass: if both current and previous are goal, allow staying
            if (
                get_position_representation(problem, (last_i, last_j)) == expected_goal
                and get_position_representation(problem, current_pos) == expected_goal
            ):
                last_pos = (i, j, t)
                continue

            # Valid move check using shared validation utility
            if not is_valid_move(problem, (last_i, last_j), current_pos):
                result["valid"] = False
                result["reason"] = "invalid_move"
                result["message"] = (
                    f"❌ Robot {robot_num}: Invalid move from ({last_i}, {last_j}, {last_t}) to ({i}, {j}, {t})"
                )
                result["details"]["invalid_moves"] = [(last_i, last_j, last_t, i, j, t)]
                return result
        last_pos = (i, j, t)

    # 7. Final check: Goal times
    goal_times = [
        t
        for i, j, t in positions
        if get_position_representation(problem, (i, j)) == expected_goal
    ]
    result["details"]["goal_times"] = goal_times

    result["valid"] = True
    result["reason"] = "valid_path"
    result["message"] = f"✅ Robot {robot_num}: Solution is valid"
    return result


def _get_notation_abstraction(problem):
    """Get the appropriate notation abstraction for the problem type."""
    problem_type = problem.get_format_type()

    if problem_type == "grid":
        return GridNotation()
    elif problem_type in ["graph", "both"]:
        return GraphNotation()
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")


class BaseNotation:
    """Base notation abstraction with common methods."""

    def get_expected_time_range(self, problem, robot_id, T):
        """Get expected time range for robot."""
        robot = problem.robots[robot_id]
        return set(range(robot.start_time, T))

    def get_start_time(self, problem, robot_id):
        """Get start time for robot."""
        robot = problem.robots[robot_id]
        return robot.start_time


class GridNotation(BaseNotation):
    """Notation abstraction for grid problems."""

    def get_start_position(self, problem, robot_id):
        """Get start position for robot."""
        robot = problem.robots[robot_id]
        return robot.start

    def get_goal_position(self, problem, robot_id):
        """Get goal position for robot."""
        robot = problem.robots[robot_id]
        return robot.goal

    def has_obstacle_check(self):
        """Whether this notation checks for obstacles."""
        return True

    def get_obstacles(self, problem):
        """Get obstacles for the problem."""
        return problem.grid.obstacles


class GraphNotation(BaseNotation):
    """Notation abstraction for graph problems."""

    def get_start_position(self, problem, robot_id):
        """Get start node for robot."""
        robot = problem.robots[robot_id]
        start_node = (
            robot.start
            if isinstance(robot.start, int)
            else problem.graph.get_node_from_position(robot.start)
        )
        return start_node

    def get_goal_position(self, problem, robot_id):
        """Get goal node for robot."""
        _, goal_node = problem.get_graph_robot_current_goal(robot_id)
        return goal_node

    def has_obstacle_check(self):
        """Whether this notation checks for obstacles (graphs don't)."""
        return False

    def get_obstacles(self, problem):
        """Get obstacles for the problem (not used for graphs)."""
        return set()


def convert_tuple_keys_to_str(obj):
    """
    Recursively convert tuple keys in dictionaries to strings.
    This is useful for serializing dictionaries with tuple keys to JSON.
    """
    if isinstance(obj, dict):
        return {
            str(k) if isinstance(k, tuple) else k: convert_tuple_keys_to_str(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_tuple_keys_to_str(i) for i in obj]
    else:
        return obj
