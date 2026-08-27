import pennylane as qml
import numpy as np
from abc import ABC, abstractmethod
from quantum.utils import paths
from quantum.utils.validation import is_valid_move
from quantum.utils.logger import get_logger
from quantum.utils import preprocess as preprocess_modes
from collections import Counter


class BaseQUBO(ABC):
    """
    Base class for QUBO builders that provides shared fields and common
    utilities. Subclasses should implement build() and may override any
    inherited methods as needed.

    Required attributes on the provided problem instance:
    - grid or graph structure as required by the subclass
    - start, end (optional depending on subclass)
    - T (total time horizon if using windowed approach)
    """

    def __init__(
        self,
        problem,
        penalties,
        name="unnamed",
        var_limit=601,
        window_max_steps=None,
        distance_scaling=None,
        robot_window_limits=None,
        verbose_level=2,
        log_reductions=True,
    ):
        self.problem = problem
        self.penalties = penalties
        self.name = name
        self.var_limit = var_limit
        self.log_reductions = log_reductions
        # Per-robot window limits: dict {robot_id: max_steps}
        # If a robot is not in this dict, no specific limit is applied
        self.robot_window_limits = robot_window_limits or {}
        # Windowing/time slicing support
        self.current_T = 0  # This represents the current global time in the problem
        self.total_t = getattr(self.problem, "T", 1)
        self.t_max = window_max_steps or self.max_window_size()
        self.iter = 0
        # QUBO dict
        self.Q = {}
        self.initial_num_vars = 0  # To be set by subclass during build
        # Optional knob used by grid subclass
        self.distance_scaling = distance_scaling
        self.prev_solution = []
        self.verbose_level = verbose_level
        # Initialize logger
        self.logger = get_logger()  # Use global logger level
        self._warn_if_horizon_too_tight()
        # Populated by get_fixed_variables() before build(); keyed by (robot_id, t_window_relative)
        self._active_cells = None
        self._warned_unrestricted_build = False

    def _warn_if_horizon_too_tight(self):
        """
        Heads-up only, not a hard error: unlike ILP/CBS -- which solve the
        whole horizon in one shot and reject this outright, see
        quantum.builder.ILPBuilder.validate_time_horizon() -- QUBO/QAOA
        windowing means a robot's T below what a clean path needs is a
        legitimate input, not just a mistake: it's exactly what you'd set
        to study how windowed reduction/approximation degrades under a
        tight budget. So just make sure it's a choice the caller is aware
        of, rather than something that only shows up later as an
        unreached-goal benchmark result.
        """
        if self.problem.grid is None:
            return
        for robot in self.problem.robots.values():
            dist = self.problem.manhattan_distance(robot.start, robot.goal)
            if robot.T - 1 < dist:
                self.logger.minimal(
                    f"[WARNING] Robot '{robot.robot_id}' has T={robot.T} "
                    f"(={robot.T - 1} moves) but needs at least {dist} "
                    f"(Manhattan distance) to reach its goal from its start -- "
                    f"this horizon may not be enough."
                )

    # Subclasses must implement build to populate self.Q
    @abstractmethod
    def build(self, constraints_to_apply=None):
        """Build the QUBO dictionary for the current window and return it."""
        raise NotImplementedError

    # Cells/nodes above which an unrestricted build is worth flagging — cheap
    # on small synthetic maps, but apply_one_hot()'s per-timestep constraint
    # loop is O(state_space²), so above this a silent full-grid/full-graph
    # fallback turns a windowed solve into a multi-second-to-minutes stall.
    _UNRESTRICTED_BUILD_WARN_THRESHOLD = 200

    def _warn_if_unrestricted_build(self, state_space_size):
        """
        Log once (at `minimal`, so it's visible without high verbosity) when
        build() is about to run over the full grid/graph instead of a
        BFS-reduced window — i.e. _active_cells hasn't been populated yet.
        Expected for the deliberate preprocess=False debug path; anywhere
        else it means _prepare_window() didn't run first as it should have
        (see update_problem()'s comment for the bug this once caused).
        """
        if (
            self._active_cells is None
            and not self._warned_unrestricted_build
            and state_space_size > self._UNRESTRICTED_BUILD_WARN_THRESHOLD
        ):
            self._warned_unrestricted_build = True
            self.logger.minimal(
                f"⚠️  Building an unrestricted QUBO over all {state_space_size} "
                "cells/nodes (no BFS-reduced window active). Expected only "
                "under preprocess=False — otherwise _active_cells should "
                "have been populated first, and this is an O(n²) footgun."
            )

    # Shared: QUBO -> Ising mapping (identical across formats)
    def qubo_to_ising(self):
        """
        Convert the QUBO (upper triangle) dictionary to an Ising Hamiltonian.
        Returns (qml.Hamiltonian, constant_offset).
        """
        linear_coeffs = {}
        quadratic_terms = {}
        constant = 0.0
        for (i, j), qij in self.Q.items():
            if i == j:
                constant += qij / 2
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 2
            else:
                constant += qij / 4
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 4
                linear_coeffs[j] = linear_coeffs.get(j, 0) - qij / 4
                quadratic_terms[i, j] = quadratic_terms.get((i, j), 0) + qij / 4

        coeffs = []
        observables = []
        for i in sorted(linear_coeffs):
            if linear_coeffs[i] != 0:
                coeffs.append(linear_coeffs[i])
                observables.append(qml.PauliZ(i))
        for (i, j), val in quadratic_terms.items():
            if val != 0:
                coeffs.append(val)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
        Hc = qml.Hamiltonian(coeffs, observables)
        return Hc, constant

    def get_wires(self):
        """
        Return set of unique variable indices in the current QUBO.
        """
        if not self.Q:
            self.logger.standard("QUBO dictionary is empty. Build the QUBO first.")
            return set()
        qubit_indices = set()
        for i, j in self.Q.keys():
            qubit_indices.update([i, j])
        return qubit_indices

    # Shared: count wires from Q
    def get_num_wires(self):
        """
        Return the number of unique variable indices in the current QUBO.
        """
        return len(self.get_wires())

    # Shared: compute max window size based on var_limit
    def max_window_size(self):
        """
        Estimate the maximum window size (in timesteps) before the QUBO would
        exceed var_limit variables.

        Uses BFS-bounded reachability: at window step s, at most connectivity^s
        cells are reachable per robot (capped at total free cells). Grid uses
        4-connectivity; graph uses max node degree. This is tighter than flat
        M×N / num_nodes estimates.

        Also respects per-robot window limits and enforces a minimum of 2 so
        current_T always advances. Real feasibility is validated in
        _prepare_window() after get_logical_variables() gives true counts.
        """
        fmt = self.problem.get_format_type()

        if fmt == "graph":
            # Use max degree as connectivity estimate (analogous to grid's 4-connectivity)
            connectivity = max(
                (len(self.graph.adjacency.get(n, [])) for n in range(self.num_nodes)),
                default=1,
            )
            max_free_cells = self.num_nodes
        else:  # grid
            connectivity = len(
                self.problem.grid.moves
            )  # 4 for standard 4-connected grid
            max_free_cells = self.problem.grid.M * self.problem.grid.N - len(
                self.problem.grid.obstacles
            )

        robot_active_timeline = self.problem.get_robot_per_timestep()

        min_robot_limit = float("inf")
        for robot_id, limit in self.robot_window_limits.items():
            if self.problem.robots[robot_id].active:
                min_robot_limit = min(min_robot_limit, limit)

        needed_vars = 0
        result = None
        flooded = False  # once connectivity^step >= max_free_cells, stay flat

        for t in range(self.current_T, self.total_t):
            step = t - self.current_T

            if not flooded:
                bfs_raw = connectivity**step
                if bfs_raw >= max_free_cells:
                    bfs_raw = max_free_cells
                    flooded = True
                vars_at_step = bfs_raw
            else:
                vars_at_step = max_free_cells

            active_robots_at_t = [
                r
                for r in robot_active_timeline.get(t, [])
                if self.problem.robots[r].active
            ]
            needed_vars += len(active_robots_at_t) * vars_at_step

            window_size = t - self.current_T

            if needed_vars > self.var_limit:
                result = window_size
                break

            if window_size >= min_robot_limit:
                result = min_robot_limit
                break

        if result is None:
            # +1 so the final window advances current_T past total_t and exits the loop
            max_possible = self.total_t - self.current_T + 1
            result = (
                min(max_possible, min_robot_limit)
                if min_robot_limit != float("inf")
                else max_possible
            )

        # Enforce minimum 2-step window to avoid current_T stalling.
        # Real feasibility (var_limit vs actual sparse vars) is validated in
        # _prepare_window() after get_logical_variables() gives true counts.
        if result < 2 and (self.total_t - self.current_T) > 0:
            result = 2

        return result

    def get_active_robot_in_window(self):
        """
        Get list of robot IDs that are active in the current window.
        They are sorted by priority in descending order.
        Filters out robots that have reached their goal (active=False).
        """
        active_robots = set()
        robot_active_timeline = self.problem.get_robot_per_timestep()
        for t in range(self.current_T, self.current_T + self.t_max):
            for robot_id in robot_active_timeline.get(t, []):
                # Only include robots that are still active (haven't reached goal)
                if self.problem.robots[robot_id].active:
                    active_robots.add(robot_id)
        # (-priority, robot_id): descending priority, then robot_id as a total
        # tiebreak. Sorting on priority alone left equal-priority robots in
        # active_robots' set-iteration order, which is PYTHONHASHSEED-dependent
        # -- so the diagonal fixer processed them in a different order per
        # process, the first robot won any shared corridor, and the whole
        # windowed solve came out valid or invalid purely by hash seed.
        return sorted(
            active_robots, key=lambda x: (-self.problem.robots[x].priority, x)
        )

    def get_active_robots_per_timestep_in_window(self):
        """
        Get a dict mapping each timestep in the current window
        to the list of active robot IDs at that timestep.
        Filters out robots that have reached their goal (active=False).
        """
        robot_active_timeline = self.problem.get_robot_per_timestep()
        active_per_timestep = {}

        for t in range(self.current_T, self.current_T + self.t_max):
            active_robots = robot_active_timeline.get(t, [])
            # Filter to only include robots that are still active
            active_robots = [r for r in active_robots if self.problem.robots[r].active]
            if active_robots:  # only include if something active
                active_per_timestep[t] = active_robots

        return active_per_timestep

    # Shared: window update/reset
    def update_problem(self, solution=[]):
        """Advance window and optionally update start state for next build."""
        self.iter += 1  # Now iter is only used for information
        # The new method to keep time is current_T which to keep track of overlapping first step with last step
        # It simply does t_max - 1 (meaning that if you only can render one time step you will always be in that time)
        # Because you need to render the step where you were and the following iteration (qubo renders the transition)
        # Note that this also helps debug since now the global time don't care about iterations (which clipped time)
        self.logger.standard(
            f"🔄 Adjusting window: current_T from {self.current_T} to {self.current_T + self.t_max - 1}"
        )
        self.current_T += self.t_max - 1
        # We update current time and recalculate t_max for next window
        self.t_max = self.max_window_size()
        if self.t_max > 0:
            for robot_num, new_segment in solution.items():
                robot_id = list(self.problem.robots.keys())[robot_num]
                robot = self.problem.robots[robot_id]
                old_path = robot.path
                merged = paths.merge_paths(old_path, new_segment)

                robot.path = merged
                robot.current_position = merged[-1][:2]

                # Early stop: Check if robot has reached its goal
                if robot.is_at_goal() and robot.active:
                    self.logger.standard(
                        f"Robot {robot_id} reached goal at position {robot.current_position}. Marking as inactive."
                    )
                    robot.active = False

            # Recalculate window size after marking robots inactive
            # This allows immediate benefit from larger windows when robots finish
            new_t_max = self.max_window_size()
            if new_t_max > self.t_max:
                self.logger.standard(
                    f"Window size increased from {self.t_max} to {new_t_max} after robots became inactive"
                )
                self.t_max = new_t_max

            # Clear stale BFS data — solve() will repopulate before next build.
            # Deliberately NOT calling self.build() here: with _active_cells
            # cleared, it would fall back to a full-grid build (_cells()'s
            # unreduced path), which _prepare_window()'s own build() call
            # immediately discards once it repopulates _active_cells — pure
            # waste that's invisible on tiny synthetic maps but dominates
            # runtime on real-sized grids. The preprocess=False raw loops
            # (PennylaneSolver/DWaveSolver) that read builder.Q without going
            # through _prepare_window() call builder.build() themselves right
            # after update_problem() for this reason.
            self._active_cells = None

    def reset_problem(self):
        """Reset windowing and restore initial start position if available."""
        for robot in self.problem.robots.values():
            robot.reset()
        self.iter = 0
        self.current_T = 0
        self.t_max = self.max_window_size()

    # Shared utilities for Q manipulation
    def dict_to_array(self, fill_value=0):
        if not self.Q:
            return np.array([[]])
        rows, cols = zip(*self.Q.keys())
        shape = (max(rows) + 1, max(cols) + 1)
        arr = np.full(shape, fill_value, dtype=float)
        for (r, c), val in self.Q.items():
            arr[r, c] = val
        return arr

    def reduce_qubo(self, fixed_vars, log_reductions=False):
        """
        Reduce a QUBO dictionary by fixing variables.
        Assumes Q uses upper triangular form: (i,j) with i <= j only.

        Args:
            Q: QUBO dictionary
            fixed_vars: dict {idx: value} or numpy array
            log_reductions: If True, track all changes for potential reversal (slower)

        Returns:
            (reduced_Q, const_offset, reduction_log)
        """
        if isinstance(fixed_vars, np.ndarray):
            fixed_dict = {
                i: int(v) for i, v in enumerate(fixed_vars) if not np.isnan(v)
            }
        else:
            fixed_dict = fixed_vars

        # Use set for O(1) lookups
        fixed_set = set(fixed_dict.keys())

        reduced_Q = {}
        const_offset = 0
        reduction_log = [] if log_reductions else None

        for key, coeff in self.Q.items():
            i, j = key
            i_fixed = i in fixed_set
            j_fixed = j in fixed_set

            if not i_fixed and not j_fixed:
                # Both free: keep as is
                reduced_Q[key] = coeff
            elif i_fixed and j_fixed:
                # Both fixed: add to constant
                const_offset += coeff * fixed_dict[i] * fixed_dict[j]
                if log_reductions:
                    reduction_log.append(
                        {
                            "type": "both_fixed",
                            "original_key": key,
                            "coeff": coeff,
                            "fixed_values": (fixed_dict[i], fixed_dict[j]),
                        }
                    )
            elif i_fixed:
                # Only i fixed
                if fixed_dict[i] == 1:
                    reduced_Q[(j, j)] = reduced_Q.get((j, j), 0) + coeff
                    if log_reductions:
                        reduction_log.append(
                            {
                                "type": "i_fixed",
                                "original_key": key,
                                "coeff": coeff,
                                "fixed_var": i,
                                "fixed_value": fixed_dict[i],
                                "free_var": j,
                            }
                        )
            else:  # j_fixed
                # Only j fixed
                if fixed_dict[j] == 1:
                    reduced_Q[(i, i)] = reduced_Q.get((i, i), 0) + coeff
                    if log_reductions:
                        reduction_log.append(
                            {
                                "type": "j_fixed",
                                "original_key": key,
                                "coeff": coeff,
                                "fixed_var": j,
                                "fixed_value": fixed_dict[j],
                                "free_var": i,
                            }
                        )

        return reduced_Q, const_offset, reduction_log or []

    def reverse_reduction(self, reduced_Q, reduction_log, var_to_unfix):
        """
        Reverse the reduction for a specific variable.

        Args:
            reduced_Q: The reduced QUBO dictionary
            reduction_log: Log from reduce_qubo
            var_to_unfix: Variable index to unfix

        Returns:
            (restored_Q, removed_const_offset)
        """
        restored_Q = reduced_Q.copy()
        removed_const = 0

        for entry in reduction_log:
            orig_key = entry["original_key"]
            i, j = orig_key
            coeff = entry["coeff"]

            if entry["type"] == "both_fixed":
                # Check if either variable is the one we're unfixing
                if i == var_to_unfix or j == var_to_unfix:
                    # Remove from constant
                    fixed_vals = entry["fixed_values"]
                    removed_const += coeff * fixed_vals[0] * fixed_vals[1]

                    # If only one is being unfixed, convert to linear term
                    if i == var_to_unfix and j != var_to_unfix:
                        # j is still fixed
                        if fixed_vals[1] == 1:
                            restored_Q[(i, i)] = restored_Q.get((i, i), 0) + coeff
                    elif j == var_to_unfix and i != var_to_unfix:
                        # i is still fixed
                        if fixed_vals[0] == 1:
                            restored_Q[(j, j)] = restored_Q.get((j, j), 0) + coeff
                    else:
                        # Both being unfixed, restore original term
                        restored_Q[orig_key] = coeff

            elif entry["type"] == "i_fixed":
                # i was fixed, j was free
                if entry["fixed_var"] == var_to_unfix:
                    free_var = entry["free_var"]
                    # Remove the linear term we added
                    if entry["fixed_value"] == 1:
                        restored_Q[(free_var, free_var)] -= coeff
                        if abs(restored_Q[(free_var, free_var)]) < 1e-10:
                            del restored_Q[(free_var, free_var)]
                    # Restore original interaction term
                    restored_Q[orig_key] = coeff

            elif entry["type"] == "j_fixed":
                # j was fixed, i was free
                if entry["fixed_var"] == var_to_unfix:
                    free_var = entry["free_var"]
                    # Remove the linear term we added
                    if entry["fixed_value"] == 1:
                        restored_Q[(free_var, free_var)] -= coeff
                        if abs(restored_Q[(free_var, free_var)]) < 1e-10:
                            del restored_Q[(free_var, free_var)]
                    # Restore original interaction term
                    restored_Q[orig_key] = coeff

        return restored_Q, removed_const

    def reduce_diag_fixed_vars_iterative(self):
        """
        Iteratively apply diag_fixed_vars until no new fixed variables are found.

        diag_fixed_vars() applies every reduction to self.Q internally (via
        _process_robot_timesteps and _handle_bfs_recalculation), so no outer
        reduce_qubo call is needed here. self.reduction_log is maintained during
        the run so that reverse_reduction can undo fixings when BFS recalculation
        is triggered, then cleared on exit.

        prior_fixed carries the running total across passes: a robot fixed to
        1 in an earlier pass has its own (i,i) entry folded away by reduce_qubo,
        so a later pass's cross-robot collision check
        (_collides_with_fixed_rival) can't see it unless we hand back the
        accumulated positions explicitly -- otherwise a collision correctly
        deferred in pass N gets blindly re-picked in pass N+1, once the rival
        occupying it is no longer visible in Q.

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        self.reduction_log = []
        total_fixed = {}
        while True:
            new_fixed = self.diag_fixed_vars(prior_fixed=total_fixed)
            if not new_fixed:
                break
            total_fixed.update(new_fixed)
        self.reduction_log = []  # Memory cleanup
        return total_fixed

    def list_to_dict_solution(self, solution_list):
        if isinstance(solution_list, list) and len(solution_list) > 0:
            solution_dict = (
                solution_list[0] if isinstance(solution_list[0], dict) else {}
            )
        elif isinstance(solution_list, dict):
            solution_dict = solution_list
        else:
            solution_dict = {}
        return solution_dict

    def reconstruct_solution(self, reduced_sol, fixed_vars, total_vars):
        if isinstance(reduced_sol, list):
            reduced_sol = self.list_to_dict_solution(reduced_sol)
        full_sol_dict = {}
        for i in range(total_vars):
            if i in fixed_vars:
                full_sol_dict[i] = np.int8(fixed_vars[i])
            elif i in reduced_sol:
                full_sol_dict[i] = np.int8(reduced_sol[i])
            else:
                full_sol_dict[i] = np.int8(0)
        return full_sol_dict

    def reachable_positions(self, robot, start_time, end_time):
        """
        Compute reachable positions per time step with backtracking.

        Args:
            robot: Robot object with current_position and robot_num.
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """
        pass

    def reachable_positions_aggressive(
        self, robot, start, start_time, end_time, blocked=None, allow_wait_at=None
    ):
        """
        Compute reachable positions per time step without backtracking.
        That means once a cell is reached, it won't be revisited in future time steps.

        Args:
            robot: Robot object with current_position and robot_num.
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).
            blocked: optional {t: {cell, ...}} of cells/nodes to exclude at
                that specific timestep only, as if temporary obstacles --
                see subclass implementations for how it's used during
                collision-driven recalculation.
            allow_wait_at: optional set of timesteps at which the robot may
                stay put instead of the walk ending early when blocking/
                no-revisit leaves no forward candidates -- a targeted,
                one-step opt-in to reachable_positions_safe()'s semantics.

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """
        pass

    @abstractmethod
    def reachable_positions_safe(self, robot, start, start_time, end_time):
        """
        Monotone reachability: the set at t is the set at t-1 unioned with its
        neighbours, so staying in place and revisiting are both allowed.

        This is the same semantics as ILPBuilder.bfs_reachable_sets(), and it
        is the only variant here that cannot exclude a feasible solution --
        reachable_positions_aggressive() forces a brand-new cell every
        timestep, which forbids the waiting that multi-robot yielding depends
        on. Costs roughly 10-19x the variables; see quantum.utils.preprocess.

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} per time step.
        """
        pass

    def reachable_for_window(self, robot, start, start_time, end_time, variant=None):
        """Dispatch to the reachability policy `variant` names.

        The variant is a parameter rather than builder state so that both
        halves of a pre-processing mode -- which BFS, and whether to run the
        numerical stage -- arrive the same way, through
        BaseSolver._prepare_window(). Storing one of them on the builder made
        its behaviour depend on which solve() ran last.

        Defaults to the aggressive policy so a builder driven directly (a
        script, a test) behaves as it did before the modes existed.
        """
        if variant == preprocess_modes.BFS_VARIANT_SAFE:
            return self.reachable_positions_safe(robot, start, start_time, end_time)
        return self.reachable_positions_aggressive(robot, start, start_time, end_time)

    # Decided to refactor into multiple functions
    # Easier debugging and previous function had troubles with breaks and variable
    def diag_fixed_vars(self, prior_fixed=None):
        """
        Identify and fix variables based on diagonal coefficients with adjacency validation.

        Args:
            prior_fixed: {var_idx: value} accumulated by earlier passes of
                reduce_diag_fixed_vars_iterative(). Seeded into this pass's
                total_fixed (but not returned) so _collides_with_fixed_rival
                can still see a rival's position after that rival's own Q
                entry has been folded away -- otherwise a collision correctly
                deferred in one pass gets blindly re-picked in the next, once
                the occupying rival is no longer visible in Q.

        Returns:
            dict: {variable_index: fixed_value} newly fixed in this pass only
                (fixed_value is 0 or 1)
        """
        total_fixed = dict(prior_fixed or {})
        seed_keys = set(total_fixed.keys())
        n = self.initial_num_vars
        type = self.problem.get_format_type()

        # Setup based on problem type
        if type == "grid":
            M, N = self.problem.grid.M, self.problem.grid.N
            vars_per_time = M * N
            adjacency_dict = self.problem.grid.adjacency
        else:  # graph
            vars_per_time = self.num_nodes
            adjacency_dict = self.problem.graph.adjacency

        robot_nums = self.problem.get_robot_nums()

        # Process each robot sequentially
        for robot_id in self.get_active_robot_in_window():
            self._process_robot_timesteps(
                robot_id,
                robot_nums,
                n,
                type,
                vars_per_time,
                adjacency_dict,
                M if type == "grid" else None,
                N if type == "grid" else None,
                total_fixed,
            )

        return {k: v for k, v in total_fixed.items() if k not in seed_keys}

    def _process_robot_timesteps(
        self,
        robot_id,
        robot_nums,
        n,
        type,
        vars_per_time,
        adjacency_dict,
        M,
        N,
        total_fixed,
    ):
        """Process all timesteps for a single robot."""
        robot_num = robot_nums[robot_id]
        robot_offset = robot_num * (vars_per_time * self.total_t)
        goal = self.problem.robots[robot_id].goal

        prev_fixed_pos = None
        prev_timestep = None

        # Collect initial timestep variables
        timestep_vars = self._collect_timestep_vars(robot_offset, vars_per_time, n)

        # Process timesteps in order
        for t in sorted(timestep_vars.keys()):
            # Refresh timestep_vars after each iteration since Q changes
            timestep_vars = self._collect_timestep_vars(robot_offset, vars_per_time, n)
            # self.logger.debug(f"Timestep {t}, robot {robot_id}: {timestep_vars}")

            if t not in timestep_vars:
                continue

            vars_at_t = timestep_vars[t]

            # Check if we need adjacency validation
            should_validate = (
                prev_fixed_pos is not None
                and prev_timestep is not None
                and t == prev_timestep + 1
            )

            # Determine what to fix at this timestep
            fix_result = self._determine_fixes_for_timestep(
                vars_at_t,
                should_validate,
                prev_fixed_pos,
                robot_id,
                t,
                type,
                goal,
                adjacency_dict,
                total_fixed=total_fixed,
                own_robot_num=robot_num,
            )

            if fix_result["needs_bfs_recalc"]:
                if prev_fixed_pos is None:
                    # No anchor to recalculate reachability from -- this fires
                    # only when the collision check trips on the very first
                    # timestep processed for this robot (e.g. its start cell
                    # collides with an already-fixed rival), which BFS
                    # recalculation can't resolve either. Leave it unfixed
                    # for the solver rather than crash on a None anchor.
                    self.logger.standard(
                        f"⚠️  Robot {robot_id} collides with an already-fixed "
                        f"rival at t={t} with no prior fixed position to "
                        "recalculate from -- leaving unfixed for the solver."
                    )
                    break
                # Recalculate reachability and fix all remaining timesteps.
                # If the trigger was a collision (not an adjacency mismatch),
                # seed the recalculation with the colliding cell blocked so
                # it's excluded from the recompute itself rather than being
                # re-offered and rejected again.
                initial_blocked = {}
                blocked_var = fix_result.get("blocked_var")
                if blocked_var is not None:
                    b_i, b_j, _, _ = paths.decode_position(blocked_var, self.problem)
                    initial_blocked = {t: {(b_i, b_j)}}
                prev_fixed_pos = self._handle_bfs_recalculation(
                    robot_id,
                    prev_fixed_pos,
                    prev_timestep,
                    t,
                    robot_offset,
                    vars_per_time,
                    type,
                    M,
                    N,
                    adjacency_dict,
                    timestep_vars,
                    total_fixed,
                    blocked=initial_blocked,
                )
                prev_timestep = t
                break  # BFS handles all remaining timesteps
            else:
                # Apply normal fixes
                if fix_result["fixes"]:
                    total_fixed.update(fix_result["fixes"])
                    self.Q, _, log = self.reduce_qubo(
                        fix_result["fixes"], log_reductions=self.log_reductions
                    )
                    self.reduction_log.extend(log)

                    # Update prev_fixed_pos to the variable fixed to 1 (if any)
                    for var_idx, val in fix_result["fixes"].items():
                        if val == 1:
                            prev_fixed_pos = var_idx
                            prev_timestep = t
                            break

    def _collect_timestep_vars(self, robot_offset, vars_per_time, n):
        """Collect current variables grouped by timestep for this robot."""
        timestep_vars = {}

        for i in range(n):
            if (i, i) in self.Q and self.Q[(i, i)] != 0:
                if robot_offset <= i < robot_offset + (vars_per_time * self.total_t):
                    local_idx = i - robot_offset
                    t = local_idx // vars_per_time
                    if t not in timestep_vars:
                        timestep_vars[t] = []
                    timestep_vars[t].append((i, self.Q[(i, i)]))

        return timestep_vars

    def _collides_with_fixed_rival(self, var_idx, t, total_fixed, own_robot_num):
        """
        True if `var_idx` (a candidate cell for the robot currently being
        processed, at window-relative timestep `t`) is already occupied by a
        *different* robot that diag-fixing has already committed to 1 earlier
        in this pass.

        Robots are processed one at a time in priority order
        (get_active_robot_in_window()), each fully fixed before the next
        starts, so any rival entry in total_fixed at this point is final for
        this pass -- there's nothing to arbitrate here, just something to
        avoid walking into.
        """
        if not total_fixed or own_robot_num is None:
            return False
        i, j, _, _ = paths.decode_position(var_idx, self.problem)
        for fixed_idx, val in total_fixed.items():
            if val != 1:
                continue
            fi, fj, ft, frn = paths.decode_position(fixed_idx, self.problem)
            if frn != own_robot_num and ft == t and (fi, fj) == (i, j):
                return True
        return False

    def _collides_with_swap_rival(
        self, var_idx, t, prev_fixed_pos, total_fixed, own_robot_num
    ):
        """
        True if picking `var_idx` at window-relative `t` would swap places
        with a different, already-committed robot: that rival occupied
        var_idx's cell at t-1, and is *also* fixed to move into this robot's
        own t-1 cell at t -- both crossing through each other's edge at the
        same transition. _collides_with_fixed_rival only catches same-
        cell/same-time (vertex) collisions; this is the other half K_swap is
        supposed to cover but never gets a say in, same reason as the vertex
        case (see _collides_with_fixed_rival's docstring).
        """
        if not total_fixed or own_robot_num is None or prev_fixed_pos is None:
            return False
        cand_i, cand_j, _, _ = paths.decode_position(var_idx, self.problem)
        own_prev_i, own_prev_j, _, _ = paths.decode_position(
            prev_fixed_pos, self.problem
        )
        t_prev = t - 1
        for fixed_idx, val in total_fixed.items():
            if val != 1:
                continue
            fi, fj, ft, frn = paths.decode_position(fixed_idx, self.problem)
            if frn == own_robot_num or ft != t_prev or (fi, fj) != (cand_i, cand_j):
                continue
            # Rival was at our candidate's cell one step ago -- check it's
            # also fixed to move into our own previous cell at t.
            for fixed_idx2, val2 in total_fixed.items():
                if val2 != 1:
                    continue
                fi2, fj2, ft2, frn2 = paths.decode_position(fixed_idx2, self.problem)
                if (
                    frn2 == frn
                    and ft2 == t
                    and (fi2, fj2) == (own_prev_i, own_prev_j)
                ):
                    return True
        return False

    def _determine_fixes_for_timestep(
        self,
        vars_at_t,
        should_validate=False,
        prev_fixed_pos=None,
        robot_id=None,
        t=None,
        type=None,
        goal=None,
        adjacency_dict=None,
        total_fixed=None,
        own_robot_num=None,
    ):
        """
        Determine which variables to fix at this timestep.

        Returns:
            dict with keys:
                - 'fixes': {var_idx: value} to apply
                - 'needs_bfs_recalc': bool indicating if BFS recalculation is needed
        """
        result = {"fixes": {}, "needs_bfs_recalc": False}

        # Single variable case
        if len(vars_at_t) == 1:
            var_idx = vars_at_t[0][0]

            if should_validate:
                prev_i, prev_j, _, _ = paths.decode_position(
                    prev_fixed_pos, self.problem
                )
                curr_i, curr_j, _, _ = paths.decode_position(var_idx, self.problem)
                if not is_valid_move(
                    self.problem, (prev_i, prev_j), (curr_i, curr_j), goal
                ):
                    self.logger.debug(
                        f"Variable {var_idx} not adjacent, needs BFS recalc"
                    )
                    result["needs_bfs_recalc"] = True
                    return result

            if self._collides_with_fixed_rival(var_idx, t, total_fixed, own_robot_num):
                self.logger.debug(
                    f"Variable {var_idx} collides with an already-fixed rival "
                    f"at t={t}, needs BFS recalc"
                )
                result["needs_bfs_recalc"] = True
                result["blocked_var"] = var_idx
                return result

            if self._collides_with_swap_rival(
                var_idx, t, prev_fixed_pos, total_fixed, own_robot_num
            ):
                self.logger.debug(
                    f"Variable {var_idx} would swap with an already-fixed "
                    f"rival at t={t}, needs BFS recalc"
                )
                result["needs_bfs_recalc"] = True
                result["blocked_var"] = var_idx
                return result

            result["fixes"][var_idx] = 1
            self.logger.debug(f"  Fixed {var_idx} to 1 (only reachable variable)")
            return result

        # Multiple variables - check coefficient distribution
        coeffs = [coeff for _, coeff in vars_at_t]
        counts = Counter(coeffs)

        if len(counts) <= 1:
            return result  # All same, skip

        # Find unique minimum coefficient
        min_coeff = min(counts)
        if counts[min_coeff] == 1:
            for var_idx, coeff in vars_at_t:
                if coeff == min_coeff:
                    # Check adjacency if needed
                    if should_validate:
                        prev_i, prev_j, _, _ = paths.decode_position(
                            prev_fixed_pos, self.problem
                        )
                        curr_i, curr_j, _, _ = paths.decode_position(
                            var_idx, self.problem
                        )
                        # Should I consider it a tuning problem by my side?
                        if not is_valid_move(
                            self.problem, (prev_i, prev_j), (curr_i, curr_j), goal
                        ):
                            self.logger.debug(
                                f"Variable {var_idx} (min coeff) not adjacent, needs BFS recalc"
                            )
                            result["needs_bfs_recalc"] = True
                            return result

                    if self._collides_with_fixed_rival(
                        var_idx, t, total_fixed, own_robot_num
                    ):
                        self.logger.debug(
                            f"Variable {var_idx} (min coeff) collides with an "
                            f"already-fixed rival at t={t}, needs BFS recalc"
                        )
                        result["needs_bfs_recalc"] = True
                        result["blocked_var"] = var_idx
                        return result

                    if self._collides_with_swap_rival(
                        var_idx, t, prev_fixed_pos, total_fixed, own_robot_num
                    ):
                        self.logger.debug(
                            f"Variable {var_idx} (min coeff) would swap with an "
                            f"already-fixed rival at t={t}, needs BFS recalc"
                        )
                        result["needs_bfs_recalc"] = True
                        result["blocked_var"] = var_idx
                        return result

                    result["fixes"][var_idx] = 1
                else:
                    result["fixes"][var_idx] = 0
        else:
            # Fix variables with maximum coefficient to 0
            max_coeff = max(counts)
            for var_idx, coeff in vars_at_t:
                if coeff == max_coeff:
                    result["fixes"][var_idx] = 0

        return result

    def _handle_bfs_recalculation(
        self,
        robot_id,
        prev_fixed_pos,
        prev_timestep,
        curr_t,
        robot_offset,
        vars_per_time,
        type,
        M,
        N,
        adjacency_dict,
        timestep_vars,
        total_fixed,
        blocked=None,
    ):
        """
        Recalculate reachable positions from the robot's last confirmed
        position and rebuild the window's QUBO around the widened candidate
        set, then fix accordingly. Returns the variable index fixed to 1 at
        curr_t.

        `blocked` (optional {t: {(i,j), ...}}) treats specific cells as
        temporary/dynamic obstacles for this recalculation -- set when the
        trigger was a detected cross-robot collision (see
        _collides_with_fixed_rival's caller), so the colliding cell is
        excluded from the recomputation itself rather than being re-offered
        and rejected again.

        A recalculated candidate may never have existed in Q: aggressive
        BFS's "no cell reused anywhere in the tree" bookkeeping can exclude
        a cell from a later timestep even though it's a genuine one-step
        neighbour of the robot's *actual*, confirmed position (this is what
        motivated this method -- see base_qubo.py's collision-check
        history). Rather than hand-computing a diagonal coefficient for just
        that one cell -- which would miss its adjacency/crash/swap terms to
        whatever comes after it -- this replaces the affected timesteps'
        entries in self._active_cells and calls self.build() again: a full
        but cheap (dict-driven, no solving) rebuild that regenerates every
        term correctly for whatever's newly in play. Everything already
        decided (`total_fixed`) is then re-folded back in via reduce_qubo to
        restore the current checkpoint.

        If even the widened set still collides, the colliding cell is added
        to `blocked` and the whole recalculation retries (bounded by the
        window size) before giving up and leaving the timestep unfixed for
        the actual solver.

        If dynamic-obstacle rerouting alone leaves nowhere to go at all --
        the aggressive (no-revisit) walk terminates before covering the
        window, because every forward cell is either an obstacle, already
        visited, or blocked -- that's a genuine dead end for pure rerouting,
        not something more blocking can fix. As a second-tier fallback, the
        robot is allowed to wait at the stall point (reachable_positions_aggressive's
        `allow_wait_at`) and normal no-revisit expansion resumes from there,
        rather than immediately giving up.
        """
        self.logger.debug(
            f"BFS recalculation for robot {robot_id} starting at timestep {curr_t}"
        )

        prev_i, prev_j, _, _ = paths.decode_position(prev_fixed_pos, self.problem)
        if type == "grid":
            start_pos = (prev_i, prev_j)
        else:
            start_pos = self.problem.graph.get_node_from_position((prev_i, prev_j))

        blocked = {t: set(cells) for t, cells in (blocked or {}).items()}
        wait_at = set()
        own_robot_num = self.problem.get_robot_nums()[robot_id]
        last_fixed_var = None

        for _attempt in range(2 * max(self.t_max, 1)):
            reachable = self.reachable_positions_aggressive(
                self.problem.robots[robot_id], start_pos, prev_timestep, self.t_max,
                blocked=blocked, allow_wait_at=wait_at,
            )
            self.logger.debug(f"Reachable positions: {reachable}")

            # Pure rerouting dead end: the walk stopped before covering the
            # window. Try once more allowing a wait at the stall point
            # before falling through to the per-timestep fixing below.
            stall_t = (max(reachable.keys()) + 1) if reachable else prev_timestep + 1
            if stall_t < self.t_max and stall_t not in wait_at:
                wait_at.add(stall_t)
                continue

            # Replace this robot's candidate set from curr_t onward with
            # what's actually reachable from its confirmed anchor, then
            # rebuild so every term is regenerated consistently.
            rebuilt = False
            for t_check, positions in reachable.items():
                if t_check < curr_t:
                    continue
                if set(self._active_cells.get((robot_id, t_check), [])) != positions:
                    # sorted(), not list(): a set's iteration order is
                    # PYTHONHASHSEED-dependent and would make BFS recalculation
                    # reroute differently per process.
                    self._active_cells[(robot_id, t_check)] = sorted(positions)
                    rebuilt = True

            if rebuilt:
                self.build()
                self.Q, _, log = self.reduce_qubo(
                    total_fixed, log_reductions=self.log_reductions
                )
                self.reduction_log = log

            retry_needed = False
            attempt_last_fixed = None

            for t_check in sorted([k for k in reachable.keys() if k >= curr_t]):
                reachable_positions = reachable[t_check]

                reachable_vars = self._get_or_create_reachable_vars(
                    t_check,
                    reachable_positions,
                    robot_offset,
                    vars_per_time,
                    type,
                    M,
                    N,
                    timestep_vars,
                    total_fixed,
                )

                fix_result = self._determine_fixes_for_timestep(
                    vars_at_t=reachable_vars,
                    t=t_check,
                    total_fixed=total_fixed,
                    own_robot_num=own_robot_num,
                )
                if fix_result["fixes"]:
                    total_fixed.update(fix_result["fixes"])
                    self.Q, _, log = self.reduce_qubo(
                        fix_result["fixes"], log_reductions=self.log_reductions
                    )
                    self.reduction_log.extend(log)

                    if t_check == curr_t:
                        for var_idx, val in fix_result["fixes"].items():
                            if val == 1:
                                attempt_last_fixed = var_idx
                                break
                elif fix_result["needs_bfs_recalc"]:
                    blocked_var = fix_result.get("blocked_var")
                    if blocked_var is not None:
                        b_i, b_j, _, _ = paths.decode_position(
                            blocked_var, self.problem
                        )
                        blocked.setdefault(t_check, set()).add((b_i, b_j))
                        retry_needed = True
                        break
                    # Adjacency-mismatch dead end (not a collision to route
                    # around) -- genuinely nothing more to try here.
                    self.logger.standard(
                        f"⚠️  Robot {robot_id} has no non-colliding reachable cell "
                        f"at window t={t_check} even after BFS recalculation -- "
                        "leaving unfixed for the solver instead of forcing a collision."
                    )

            if retry_needed:
                continue
            last_fixed_var = attempt_last_fixed
            break

        return last_fixed_var

    def _get_or_create_reachable_vars(
        self,
        t,
        reachable_positions,
        robot_offset,
        vars_per_time,
        type,
        M,
        N,
        timestep_vars,
        total_fixed,
    ):
        """Get existing or unfix variables for reachable positions."""
        reachable_vars = []

        # sorted(): reachable_positions is a set; a stable order keeps the fix
        # decisions below independent of PYTHONHASHSEED.
        for pos in sorted(reachable_positions):
            # Calculate variable index for this position
            if type == "grid":
                i_pos, j_pos = pos
                local_pos_idx = i_pos * N + j_pos
            else:
                local_pos_idx = pos

            var_idx = robot_offset + t * vars_per_time + local_pos_idx

            # Try to unfix if previously fixed
            if var_idx in total_fixed:
                self.Q, _ = self.reverse_reduction(self.Q, self.reduction_log, var_idx)
                del total_fixed[var_idx]
                self.logger.debug(f"  Unfixed variable {var_idx}")

            # Add if exists in Q
            if (var_idx, var_idx) in self.Q:
                reachable_vars.append((var_idx, self.Q[(var_idx, var_idx)]))

        return reachable_vars

    def _get_position_from_var(self, var_idx, type):
        """Extract position from variable index."""
        i, j, _, _ = paths.decode_position(var_idx, self.problem)

        if type == "grid":
            return (i, j)
        else:
            return self.problem.graph.get_node_from_position((i, j))
