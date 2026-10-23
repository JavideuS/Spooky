import pennylane as qml
import numpy as np
from abc import ABC, abstractmethod
from quantum.utils import paths
from quantum.utils.logger import get_logger
from quantum.utils import preprocess as preprocess_modes
from quantum.builder.preprocessing import DiagonalFixer


class BaseQUBO(DiagonalFixer, ABC):
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
        window_sizing="estimate",
        fixer="chain",
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
        # Explicit cap on every window, honoured by fit_window_to_var_limit()
        self.window_max_steps = window_max_steps
        # How _prepare_window() sizes each window: "estimate" keeps
        # max_window_size()'s connectivity**step bound, "bfs" fits the window
        # to var_limit from the real BFS layers (fit_window_to_var_limit()).
        if window_sizing not in ("estimate", "bfs"):
            raise ValueError(
                f"window_sizing must be 'estimate' or 'bfs', got {window_sizing!r}"
            )
        self.window_sizing = window_sizing
        # Diagonal fixer's path discipline (see _process_robot_timesteps):
        # "chain" pins any step whose pins keep the robot's 1-pins connected;
        # "prefix" also stops a robot's pinning at its first undecided step.
        if fixer not in ("chain", "prefix"):
            raise ValueError(f"fixer must be 'chain' or 'prefix', got {fixer!r}")
        self.fixer = fixer
        # Windowing/time slicing support
        self.current_T = 0  # This represents the current global time in the problem
        self.total_t = getattr(self.problem, "T", 1)
        self.t_max = window_max_steps or self.max_window_size()
        self.iter = 0
        # QUBO dict
        self.Q = {}
        self.initial_num_vars = 0  # To be set by subclass during build
        # Optional knob used by grid subclass
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

    def fit_window_to_var_limit(self, bfs_variant=None):
        """
        Size the current window from the real BFS layers instead of
        max_window_size()'s connectivity**step estimate, which overshoots the
        true reachable set by orders of magnitude and so leaves most of
        var_limit unused.

        A robot's BFS layers grow outward from its current cell and don't
        depend on where the window ends, so a window of W steps holds exactly
        the first W layers of a longer one. That lets one BFS over the longest
        allowed window serve every candidate W: pick the largest W whose
        logical variable count fits var_limit, then truncate to it -- no
        second BFS.

        The ceiling is the rest of the horizon (+1, as in max_window_size(),
        so the last window carries current_T past total_t), capped by
        window_max_steps and the active robots' robot_window_limits. The
        minimum stays 2 so current_T always advances; _prepare_window()
        raises if even that doesn't fit.

        Sets self.t_max and returns get_logical_variables()'s
        (fixed_ones, active_cells) for the chosen window.
        """
        ceiling = self.total_t - self.current_T + 1
        caps = [
            limit
            for robot_id, limit in self.robot_window_limits.items()
            if self.problem.robots[robot_id].active
        ]
        if self.window_max_steps:
            caps.append(self.window_max_steps)
        if caps:
            ceiling = min(ceiling, min(caps))
        self.t_max = max(ceiling, 2)

        fixed_ones, active_cells = self.get_logical_variables(bfs_variant)

        per_t = {}
        for (_, t), cells in active_cells.items():
            per_t[t] = per_t.get(t, 0) + len(cells)
        window, running = 2, 0
        for t in range(self.t_max):
            running += per_t.get(t, 0)
            if running > self.var_limit:
                break
            window = t + 1
        window = max(window, 2)
        if window >= self.t_max:
            return fixed_ones, active_cells

        self.t_max = window
        # Window-relative t of a flat index: idx // cells_per_t is
        # robot * total_t + t.
        if self.problem.get_format_type() == "graph":
            cells_per_t = self.num_nodes
        else:
            cells_per_t = self.problem.grid.M * self.problem.grid.N
        fixed_ones = {
            idx: v
            for idx, v in fixed_ones.items()
            if (idx // cells_per_t) % self.total_t < window
        }
        active_cells = {k: v for k, v in active_cells.items() if k[1] < window}
        return fixed_ones, active_cells

    def window_steps(self, robot_id):
        """(start, end) of robot_id in the current window, window-relative:
        start is 0 unless the robot starts later; end is exclusive -- its
        deadline (start_time + T) if that falls inside the window, else
        t_max."""
        robot = self.problem.robots[robot_id]
        start = max(0, robot.start_time - self.current_T)
        end_time = robot.start_time + robot.T
        if end_time > self.current_T + self.t_max:
            return start, self.t_max
        return start, end_time - self.current_T

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
