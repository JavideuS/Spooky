from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple
from quantum.utils.paths import decode_position
from quantum.utils.validation import is_valid_move
from quantum.utils.logger import get_logger


class BaseSolver(ABC):
    """
    Abstract base class for all quantum solvers.
    Defines the common interface that all solvers must implement.
    """

    def __init__(
        self,
        solver: str,
        normalize_scale: float = 0,
        num_reads: int = 10,
        max_corrections: int = 3,
        verbose_level: int = 2,
        **kwargs,
    ):
        """
        Initialize the base solver.

        Args:
            solver: Name of the solver (e.g., "dwave", "pennylane", "qiskit")
            normalize_scale: Scale factor for QUBO normalization
            num_reads: Number of reads/samples to take
            max_corrections: Maximum number of invalid move corrections to attempt
            verbose_level: Verbosity level (0=Silent, 1=Minimal, 2=Standard, 3=Debug)
            **kwargs: Additional solver-specific parameters
        """
        self.solver = solver
        self.norm_scale = normalize_scale
        self.num_reads = num_reads
        self.max_corrections = max_corrections
        self.verbose_level = verbose_level
        self.name = f"{self.solver}_reads{num_reads}"
        self._solver_params = kwargs
        self.logger = get_logger()  # Use global logger level

    # Config file already returns dict for penalties; no new constructor needed
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Create a solver instance from a configuration dictionary.

        Args:
            config: Configuration dictionary containing solver parameters

        Returns:
            Solver instance
        """
        norm_scale = config.get("normalization_scale", 0)
        num_reads = config.get("num_reads", 15)
        max_corrections = config.get("max_corrections", 3)

        # Extract solver-specific parameters
        solver_params = {
            k: v
            for k, v in config.items()
            if k
            not in ["solver", "normalization_scale", "num_reads", "max_corrections"]
        }

        # Each class expects to run its own solver
        return cls(
            normalize_scale=norm_scale,
            num_reads=num_reads,
            max_corrections=max_corrections,
            **solver_params,
        )

    def normalize_qubo(self, Q: Dict, scale: float = 1.0) -> Dict:
        """
        Normalize QUBO coefficients to a specified scale.

        Args:
            Q: QUBO dictionary
            scale: Target scale for normalization

        Returns:
            Normalized QUBO dictionary
        """
        # In case the QUBO is empty
        # It could happen in cases where the whole window gets pre-processed
        if len(Q) == 0:
            return Q
        # Extract values
        values = np.array(list(Q.values()))

        # Compute min/max
        max_val = np.max(np.abs(values))
        if max_val == 0:
            return Q

        # Scale all values to [-scale, scale]
        scale_factor = scale / max_val
        return {k: v * scale_factor for k, v in Q.items()}

    def decode_path(
        self, sample: Dict, problem, t_offset: int = 0
    ) -> List[Tuple[Tuple[int, int, int], int]]:
        """
        Decode the binary sample into a path of ((i, j, t), robot_num) tuples.
        Merges multiple time-window samples while ensuring continuity per robot.
        """
        path = []

        # Handle multiple segments (list of samples)
        if isinstance(sample, list):
            if len(sample) == 1:
                sample = sample[0]
            else:
                t_offset_running = t_offset
                path = []
                # Store last position per robot
                last_positions = {}

                for index, s in enumerate(sample):
                    sub_path = self.decode_path(s, problem, t_offset=t_offset_running)
                    if not sub_path:
                        continue

                    # Organize by robot for continuity checking
                    sub_robot_paths = self.get_robot_paths(sub_path)

                    # For each robot, check if we can clip the start
                    for robot_num, robot_path in sub_robot_paths.items():
                        if robot_num in last_positions:
                            last_pos = last_positions[robot_num]
                            first_pos = robot_path[0][:2]
                            if last_pos == first_pos:
                                # Same position continuity → remove first from subpath
                                # No time adjustment needed since we're using global timesteps via t_offset
                                sub_robot_paths[robot_num] = robot_path[1:]

                    # We have been working with dict num_robot: path
                    # So we flatten sub_robot_paths back to list of ((i,j,t), robot_num)
                    merged_sub_path = [
                        ((i, j, t), r)
                        for r, coords in sub_robot_paths.items()
                        for (i, j, t) in coords
                    ]

                    # Update last_positions
                    for (i, j, t), r in merged_sub_path:
                        last_positions[r] = (i, j)

                    # Update running time offset
                    if merged_sub_path:
                        max_t = max(x[0][2] for x in merged_sub_path)
                        t_offset_running = max_t + 1

                    path.extend(merged_sub_path)

                return path

        # Handle single dict sample
        if isinstance(sample, dict):
            qubo_type = problem.get_format_type()
            num_robots = problem.num_robots
            if qubo_type == "grid":
                M = problem.grid.M
                N = problem.grid.N
                T = problem.T
                total_vars = M * N * T * num_robots
            else:
                total_vars = len(problem.graph.nodes) * problem.T * num_robots

            for idx in range(total_vars):
                if sample.get(idx, 0) == 1:
                    i, j, t, robot_num = decode_position(idx, problem)
                    path.append(((i, j, t + t_offset), robot_num))

            return path

        return []

    def get_combined_path(
        self, path: List[Tuple[Tuple[int, int, int], int]]
    ) -> List[Tuple[int, int, int]]:
        """
        Get a combined path with all robots together, sorted by time.

        Args:
            path: List of ((i, j, t), robot_num) tuples

        Returns:
            List of (i, j, t) tuples sorted by time
        """
        if not path:
            return []

        # Sort by time, then by robot_num for consistent ordering
        sorted_path = sorted(path, key=lambda x: (x[0][2], x[1]))
        return [pos_time for (pos_time, robot_num) in sorted_path]

    def get_robot_paths(
        self, path: List[Tuple[Tuple[int, int, int], int]]
    ) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Get individual paths for each robot.

        Args:
            path: List of ((i, j, t), robot_num) tuples

        Returns:
            Dictionary mapping robot_num to list of (i, j, t) tuples
        """
        robot_paths = {}

        for (i, j, t), robot_num in path:
            if robot_num not in robot_paths:
                robot_paths[robot_num] = []
            robot_paths[robot_num].append((i, j, t))

        # Sort each robot's path by time
        for robot_num in robot_paths:
            robot_paths[robot_num].sort(key=lambda x: x[2])

        return robot_paths

    def _resolve_duplicate_timesteps(
        self, robot_paths: Dict[int, List[Tuple[int, int, int]]], problem
    ) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Resolve cases where a robot has multiple positions at the same timestep
        by choosing the one that maintains path continuity.

        Args:
            robot_paths: Dictionary mapping robot_num to list of (i, j, t) tuples
            problem: Problem instance for adjacency checking

        Returns:
            Cleaned robot_paths with duplicates resolved
        """
        cleaned_paths = {}

        for robot_num, positions in robot_paths.items():
            if not positions:
                cleaned_paths[robot_num] = []
                continue

            # Sort by time
            positions.sort(key=lambda x: x[2])

            # Group positions by timestep
            time_to_positions = {}
            for i, j, t in positions:
                if t not in time_to_positions:
                    time_to_positions[t] = []
                time_to_positions[t].append((i, j))

            # Build cleaned path
            cleaned = []
            last_pos = None

            for t in sorted(time_to_positions.keys()):
                candidates = time_to_positions[t]

                if len(candidates) == 1:
                    # No conflict, use the single position
                    chosen = candidates[0]
                else:
                    # Multiple positions at same timestep - choose based on continuity
                    if last_pos is None:
                        # No previous position, choose first candidate
                        chosen = candidates[0]
                        self.logger.standard(
                            f"⚠️  Robot {robot_num} at t={t}: Multiple positions {candidates}, "
                            f"no previous position to guide, choosing {chosen}"
                        )
                    else:
                        # Choose the position that is adjacent to the last position
                        chosen = self._choose_continuous_position(
                            last_pos, candidates, problem, robot_num, t
                        )

                cleaned.append((chosen[0], chosen[1], t))
                last_pos = chosen

            cleaned_paths[robot_num] = cleaned

        return cleaned_paths

    def _choose_continuous_position(
        self,
        last_pos: Tuple[int, int],
        candidates: List[Tuple[int, int]],
        problem,
        robot_num: int,
        t: int,
    ) -> Tuple[int, int]:
        """
        Choose the position from candidates that maintains continuity with last_pos.

        Args:
            last_pos: Previous position (i, j)
            candidates: List of candidate positions at current timestep
            problem: Problem instance for adjacency checking
            robot_num: Robot number for logging
            t: Current timestep for logging

        Returns:
            Chosen position (i, j)
        """
        problem_type = problem.get_format_type()

        # Check which candidates are adjacent to last_pos
        valid_candidates = []

        for candidate in candidates:
            if problem_type == "grid":
                # Check grid adjacency
                if candidate in problem.grid.adjacency.get(last_pos, []):
                    valid_candidates.append(candidate)
            else:
                # Check graph adjacency
                last_node = problem.graph.get_node_from_position(last_pos)
                candidate_node = problem.graph.get_node_from_position(candidate)
                if any(
                    neighbor_node == candidate_node
                    for (neighbor_node, _) in problem.graph.adjacency.get(last_node, [])
                ):
                    valid_candidates.append(candidate)

        if valid_candidates:
            chosen = valid_candidates[0]
            if len(valid_candidates) > 1:
                self.logger.debug(
                    f"⚠️  Robot {robot_num} at t={t}: Multiple valid adjacent positions "
                    f"{valid_candidates} from {last_pos}, choosing {chosen}"
                )
            else:
                self.logger.debug(
                    f"✓ Robot {robot_num} at t={t}: Resolved duplicate by continuity - "
                    f"chose {chosen} from {candidates} (adjacent to {last_pos})"
                )
            return chosen
        else:
            # No adjacent candidates - this is a discontinuity, choose first and warn
            chosen = candidates[0]
            self.logger.minimal(
                f"⚠️  Robot {robot_num} at t={t}: No adjacent position found! "
                f"Candidates {candidates} not adjacent to {last_pos}. Choosing {chosen} arbitrarily."
            )
            return chosen

    def _resolve_invalid_moves(
        self, robot_paths: Dict[int, List[Tuple[int, int, int]]], problem
    ) -> Tuple[Dict[int, List[Tuple[int, int, int]]], Dict[int, int]]:
        """
        Detect and resolve invalid (non-adjacent) moves in robot paths.
        When an invalid move is detected, truncate the path at that timestep
        and mark it for replanning.

        Args:
            robot_paths: Dictionary mapping robot_num to list of (i, j, t) tuples
            problem: Problem instance for adjacency checking

        Returns:
            Tuple of (corrected_robot_paths, invalid_moves_dict)
            where invalid_moves_dict maps robot_num to the timestep where invalid move occurred
        """
        corrected_paths = {}
        invalid_moves = {}

        for robot_num, positions in robot_paths.items():
            if not positions or len(positions) < 2:
                corrected_paths[robot_num] = positions
                continue

            # Sort by time to ensure sequential checking
            positions.sort(key=lambda x: x[2])

            # Check each consecutive pair of positions
            valid_path = [positions[0]]  # Start position is always valid

            for idx in range(1, len(positions)):
                prev_pos = (positions[idx - 1][0], positions[idx - 1][1])
                curr_pos = (positions[idx][0], positions[idx][1])
                curr_timestep = positions[idx][2]

                # Check if move is valid (adjacent or same position)
                is_valid = is_valid_move(problem, prev_pos, curr_pos)

                if is_valid:
                    valid_path.append(positions[idx])
                else:
                    # Invalid move detected - truncate path here
                    invalid_moves[robot_num] = curr_timestep
                    self.logger.standard(
                        f"❌ Robot {robot_num}: Invalid move from {prev_pos} (t={positions[idx - 1][2]}) "
                        f"to {curr_pos} (t={curr_timestep}). Truncating path and will replan from t={curr_timestep}."
                    )
                    break  # Stop processing this robot's path

            corrected_paths[robot_num] = valid_path

        return corrected_paths, invalid_moves

    def _flag_forced_collisions(self, builder, bfs_fixed, diag_fixed):
        """
        Warn when pre-processing has fixed two robots onto the same cell at
        the same absolute timestep, before K_crash/K_swap ever get a say.

        Two independent mechanisms fix variables ahead of solving:
        - bfs_fixed: get_logical_variables()'s aggressive BFS reachability
          (per-robot, no cross-robot awareness at all).
        - diag_fixed: reduce_diag_fixed_vars_iterative()'s greedy diagonal
          reduction (folds already-fixed robots' coupling terms into the
          next robot's diagonal, so it's collision-aware *except* when a
          robot has only one reachable cell left at a timestep — then it's
          forced regardless of any penalty on it).

        Tagging which stage produced each fixed variable tells you which
        gap to close: a "bfs" collision means the two robots' independent
        aggressive-BFS paths crossed; a "diag" or mixed "bfs"/"diag"
        collision means a robot was left with a single forced cell that a
        penalty could see but couldn't stop.

        Also checks fixed cells against robots that went inactive earlier,
        but only at the exact timestep they finished (robot.path[-1]) — a
        finished robot vacates its cell, it doesn't block it forever, so
        this only catches a same-instant handoff collision: an active
        robot forced into the same cell at the same t another robot was
        still occupying when it stopped being tracked. Once a robot is
        inactive it's dropped from every later window's variables entirely
        (get_active_robot_in_window), so once only one robot is left
        active, the same-window check above can never see this by itself.

        Returns:
            list of {"cell": (i, j), "time": t, "robots": [...], "sources": [...],
            "origin": "same_window" | "inactive_handoff"} — one entry per forced
            collision found, so callers (e.g. BenchmarkRunner) can cross-reference
            a run's reported conflicts against pre-processing as the root cause.
        """
        source_by_idx = {idx: "bfs" for idx, v in bfs_fixed.items() if v == 1}
        for idx, v in diag_fixed.items():
            if v == 1:
                source_by_idx[idx] = "diag"

        robot_nums = builder.problem.get_robot_nums()
        num_to_id = {num: rid for rid, num in robot_nums.items()}

        occupied = {}
        for idx, source in source_by_idx.items():
            i, j, t_window, robot_num = decode_position(idx, builder.problem)
            key = (i, j, builder.current_T + t_window)
            occupied.setdefault(key, []).append(
                (num_to_id.get(robot_num, robot_num), source)
            )

        forced_collisions = []

        for (i, j, t), robot_sources in occupied.items():
            robots = {r for r, _ in robot_sources}
            if len(robots) > 1:
                detail = ", ".join(f"{r} ({s})" for r, s in sorted(robot_sources, key=str))
                self.logger.standard(
                    f"⚠️  Pre-processing forced a collision at ({i}, {j}) t={t}: "
                    f"{detail} — fixed before solving, bypasses K_crash/K_swap."
                )
                forced_collisions.append({
                    "cell": (i, j),
                    "time": t,
                    "robots": sorted(robots, key=str),
                    "sources": sorted(robot_sources, key=str),
                    "origin": "same_window",
                })

        for robot_id, robot in builder.problem.robots.items():
            if robot.active or not robot.path:
                continue
            goal_i, goal_j, goal_t = robot.path[-1]
            robot_sources = occupied.get((goal_i, goal_j, goal_t))
            if robot_sources:
                mover = ", ".join(f"{r} ({s})" for r, s in sorted(robot_sources, key=str))
                self.logger.standard(
                    f"⚠️  Pre-processing forced {mover} into ({goal_i}, {goal_j}) "
                    f"t={goal_t}, the same cell/time {robot_id} stopped at — "
                    f"{robot_id} is inactive so it has no variable in this "
                    f"window and this bypasses K_crash/K_swap entirely."
                )
                forced_collisions.append({
                    "cell": (goal_i, goal_j),
                    "time": goal_t,
                    "robots": sorted({r for r, _ in robot_sources} | {robot_id}, key=str),
                    "sources": sorted(robot_sources, key=str) + [(robot_id, "locked_inactive")],
                    "origin": "inactive_handoff",
                })

        return forced_collisions

    def _prepare_window(self, builder):
        """
        Prepare a QUBO window: derive logical variables, build the sparse QUBO,
        and apply diagonal reduction.

        Returns:
            (fixed_vars, window_stat, is_fully_preprocessed, forced_collisions)
            - fixed_vars: {flat_idx: 1} merged with any diag-fixed variables
            - window_stat: dict with initial/final variable counts for this window
            - is_fully_preprocessed: True when the QUBO is empty after reduction
              (solver can skip sampling and go straight to _handle_iteration_result)
            - forced_collisions: list from _flag_forced_collisions() for this window
        """
        import time as timing

        t0 = timing.time()
        fixed_vars, active_cells = builder.get_logical_variables()
        builder._active_cells = active_cells
        t1 = timing.time()

        # Validate feasibility using actual sparse counts (first 2 active timesteps).
        # This is the authoritative check — BFS estimates in max_window_size() are only
        # used for sizing; here we know the real reachable cell counts.
        if (builder.total_t - builder.current_T) > 0:
            first_two_ts = sorted(set(t for (_, t) in active_cells))[:2]
            min_vars = sum(
                len(cells)
                for (_, t), cells in active_cells.items()
                if t in first_two_ts
            )
            if min_vars > builder.var_limit:
                raise ValueError(
                    f"var_limit={builder.var_limit} is too small: a minimum 2-step "
                    f"window requires {min_vars} variables based on actual reachability. "
                    f"Increase var_limit to at least {min_vars}."
                )

        builder.build()
        t2 = timing.time()

        initial_vars = builder.get_num_wires()
        bfs_fixed = fixed_vars
        diag_fixed = builder.reduce_diag_fixed_vars_iterative()
        fixed_vars = {**bfs_fixed, **diag_fixed}
        t3 = timing.time()

        forced_collisions = self._flag_forced_collisions(builder, bfs_fixed, diag_fixed)

        final_vars = builder.get_num_wires()
        vars_reduced = len(diag_fixed)
        reduction_ratio = vars_reduced / initial_vars if initial_vars > 0 else 0

        self.logger.debug(
            f"⏱️ get_logical_vars: {(t1 - t0) * 1000:.1f}ms, "
            f"build: {(t2 - t1) * 1000:.1f}ms, "
            f"diag_reduce: {(t3 - t2) * 1000:.1f}ms"
        )
        self.logger.standard(
            f"Window {builder.iter}: {initial_vars} → {final_vars} vars "
            f"(reduced {vars_reduced}, {reduction_ratio:.1%})"
        )

        window_stat = {
            "window": builder.iter,
            "initial_variables": initial_vars,
            "variables_reduced": vars_reduced,
            "final_variables": final_vars,
            "reduction_ratio": round(reduction_ratio, 4),
        }

        return fixed_vars, window_stat, final_vars == 0, forced_collisions

    @abstractmethod
    def solve_qubo(self, builder, optimization=False, preprocess=True) -> Dict[str, Any]:
        """
        Solve the QUBO problem.

        Args:
            builder: QUBOBuilder instance
            optimization: Whether to run variational parameter optimization before
                sampling (PennyLane only; ignored by classical/annealing solvers).
            preprocess: When True (default), runs _prepare_window() to apply BFS
                logical-variable reduction and diagonal fixed-var pruning, tracks
                window stats, and retries invalid moves up to max_corrections times.
                When False, runs a simpler loop with no preprocessing — useful for
                debugging the raw sampler.

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        pass

    def total_energy(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the total energy of all windows in the solution.

        Args:
            solution: Solution dictionary

        Returns:
            Total energy
        """
        return np.sum(solution["energy"])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert solver parameters to dictionary.

        Returns:
            Dictionary representation of solver parameters
        """
        result = {
            "solver": self.solver,
            "normalization_scale": self.norm_scale,
            "num_reads": self.num_reads,
            "max_corrections": self.max_corrections,
        }
        result.update(self._solver_params)
        return result

    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get solver-specific information.

        Returns:
            Dictionary with solver information
        """
        return {
            "solver": self.solver,
            "name": self.name,
            "parameters": self._solver_params,
        }

    def build_solution_from_robot_paths(self, problem) -> Dict[int, int]:
        """
        Build a solution dictionary from stored robot paths.

        This creates a binary solution dict where variable indices that are
        part of the robot paths are set to 1, and all others are 0.

        Args:
            problem: Problem instance with robots containing stored paths

        Returns:
            Dictionary mapping variable indices to binary values (0 or 1)
        """
        solution = {}

        # Get problem dimensions
        qubo_type = problem.get_format_type()
        if qubo_type == "grid":
            M = problem.grid.M
            N = problem.grid.N
            vars_per_time = M * N
        else:
            vars_per_time = len(problem.graph.nodes)

        T = problem.T
        num_robots = problem.num_robots
        total_vars = vars_per_time * T * num_robots

        # Initialize all variables to 0
        for idx in range(total_vars):
            solution[idx] = 0

        # Set variables to 1 for positions in robot paths
        for robot_num, robot_id in enumerate(problem.robots.keys()):
            robot = problem.robots[robot_id]
            robot_offset = robot_num * (vars_per_time * T)

            for i, j, t in robot.path:
                # Calculate variable index for this position
                if qubo_type == "grid":
                    local_pos_idx = i * N + j
                else:
                    # For graph, convert position to node index
                    node_idx = problem.graph.get_node_from_position((i, j))
                    local_pos_idx = node_idx

                var_idx = robot_offset + t * vars_per_time + local_pos_idx
                solution[var_idx] = 1

        return solution

    def _handle_iteration_result(self, solution, fixed_vars, builder):
        """
        Handle the result of a QUBO iteration: reconstruct solution and
        update problem.

        Args:
            solution: The solution dictionary or sample
            fixed_vars: Fixed variables from preprocessing
            builder: QUBOBuilder instance

        Returns:
            tuple: (reconstructed_solution, invalid_moves_dict)
            where invalid_moves_dict maps robot_num to timestep where invalid move occurred
        """
        # Reconstruct full solution
        full_sol = builder.reconstruct_solution(
            solution, fixed_vars, total_vars=builder.initial_num_vars
        )

        path = self.decode_path(full_sol, builder.problem, t_offset=builder.current_T)
        robot_paths = self.get_robot_paths(path)

        # Apply post-processing to resolve duplicate timesteps
        robot_paths = self._resolve_duplicate_timesteps(robot_paths, builder.problem)

        # Apply post-processing to detect and resolve invalid moves
        robot_paths, invalid_moves = self._resolve_invalid_moves(
            robot_paths, builder.problem
        )

        # print("Decoded path:", path)
        self.logger.standard("Robots paths", robot_paths)
        if invalid_moves:
            self.logger.standard(
                f"⚠️  Invalid moves detected for robots: {list(invalid_moves.keys())}"
            )

            # Don't update the problem - this will cause the solver to repeat the same window
            # The window will be rebuilt from scratch with just the start positions
            return full_sol, invalid_moves

            # SCENARIO where we want to stay in last valid cell without repeating the whole window
            # Like treating as if it was only one window from the beginning (but with smaller size)

            # self.logger.standard(f"🔄 Discarding this window and repeating from current_T={builder.current_T}")

            # Adjust builder.current_T to the earliest truncation point
            # This ensures the next window starts from where the error occurred
            # earliest_invalid_time = min(invalid_moves.values())

            # # Calculate effective t_max so that after update_problem,
            # # current_T will be at (earliest_invalid_time - 1)
            # # update_problem does: current_T += t_max - 1
            # # We want: current_T + t_max - 1 = earliest_invalid_time - 1
            # # Therefore: t_max = earliest_invalid_time - current_T

            # old_t_max = builder.t_max
            # effective_t_max = earliest_invalid_time - builder.current_T

            # print(f"🔄 Invalid move at t={earliest_invalid_time}, adjusting t_max from {old_t_max} to {effective_t_max}")
            # print(f"   After update_problem, current_T will be: {builder.current_T} + {effective_t_max} - 1 = {builder.current_T + effective_t_max - 1}")

            # # CRITICAL: Re-offset the timesteps in robot_paths to match the adjusted window
            # # The paths were decoded with the original window, but now we're changing the window size
            # # We need to adjust the timesteps so merge_paths works correctly
            # time_adjustment = old_t_max - effective_t_max
            # if time_adjustment != 0:
            #     print(f"   Re-offsetting path timesteps by -{time_adjustment} to match adjusted window")
            #     for robot_num in robot_paths:
            #         robot_paths[robot_num] = [
            #             (i, j, t - time_adjustment)
            #             for i, j, t in robot_paths[robot_num]
            #         ]

            # builder.t_max = effective_t_max

        builder.update_problem(robot_paths)
        return full_sol, invalid_moves
