import itertools
from collections import Counter, deque

import pennylane as qml
import numpy as np
from scipy.optimize import linprog
from abc import ABC, abstractmethod
from quantum.utils import paths
from quantum.utils.validation import is_valid_move
from quantum.utils.logger import get_logger


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
        encoding="one_hot",
    ):
        self.problem = problem
        self.penalties = penalties
        self.name = name
        self.var_limit = var_limit
        self.log_reductions = log_reductions
        # Variable encoding: "one_hot" (default, N vars per robot/timestep) or
        # "binary" (B = ceil(log2(N)) vars per robot/timestep). Both remain
        # selectable per-instance so they can be benchmarked against each other.
        if encoding not in ("one_hot", "binary"):
            raise ValueError(
                f"Unknown encoding '{encoding}', expected 'one_hot' or 'binary'"
            )
        self.encoding = encoding
        self.B = self._compute_bit_width()
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
        # Populated by get_fixed_variables() before build(); keyed by (robot_id, t_window_relative)
        self._active_cells = None

    # Subclasses must implement build to populate self.Q
    @abstractmethod
    def build(self, constraints_to_apply=None):
        """Build the QUBO dictionary for the current window and return it."""
        raise NotImplementedError

    def _positions_per_timestep(self):
        """Number of distinct positions a robot can occupy at one timestep:
        M*N for grid, num_nodes for graph. This is N in the one-hot encoding
        and determines B = ceil(log2(N)) for the binary encoding.
        """
        if self.problem.get_format_type() == "graph":
            return len(self.problem.graph.nodes)
        return self.problem.grid.M * self.problem.grid.N

    def _compute_bit_width(self):
        """B = ceil(log2(N)) bits needed to binary-encode N positions."""
        return paths.bit_width(self._positions_per_timestep())

    # Shared: binary global variable index for bit b of the position code at
    # (robot_num, window-relative time t). idx = r*T*B + t*B + b.
    def _binary_not_implemented(self, constraint_name):
        """Shared placeholder for binary-encoding constraints not yet
        implemented (LP max-margin fitting machinery — not built yet)."""
        raise NotImplementedError(
            f"Binary encoding: '{constraint_name}' constraint not implemented yet"
        )

    def binary_var_index(self, robot_num, t, b):
        """Global variable index for bit b of the binary-encoded position at
        (robot_num, window-relative time t). b is the bit's power of two
        (LSB-first: b=0 is worth 2**0) — see paths.bits_to_code, the single
        source of truth for this convention."""
        return robot_num * (self.total_t * self.B) + t * self.B + b

    def _all_pairs_bfs_distance(self, num_positions, neighbor_codes_fn):
        """BFS shortest-path distance (in hops) between every reachable pair
        of real position codes 0..num_positions-1, using neighbor_codes_fn(code)
        -> iterable of neighboring codes. Missing entries mean unreachable.
        """
        dist = {}
        for src in range(num_positions):
            dist[(src, src)] = 0
            frontier = deque([src])
            while frontier:
                cur = frontier.popleft()
                for nxt in neighbor_codes_fn(cur):
                    if (src, nxt) not in dist:
                        dist[(src, nxt)] = dist[(src, cur)] + 1
                        frontier.append(nxt)
        return dist

    # Which solve _fit_binary_pairwise_block dispatches to: "lp" (hard
    # max-margin separation) or "ls" (least-squares regression). Flip this
    # by hand on an instance/subclass while comparing the two — not yet
    # exposed as a constructor parameter. Default "ls": for full grid/graph
    # adjacency the LP was confirmed infeasible (no quadratic form exactly
    # separates all real-cell transitions, verified empirically — see
    # _fit_binary_pairwise_block_lp's docstring), so LP currently raises on
    # anything but toy cases. LS always returns a (possibly imperfect) fit.
    _BINARY_ADJACENCY_FIT_METHOD = "ls"

    # Set to False to test binary encoding without hard-fixing the window-
    # start position's bits (see get_logical_variables). Flip this by hand
    # while comparing, same as _BINARY_ADJACENCY_FIT_METHOD above. When True,
    # the fix is a strict win (fewer free variables, no behavior change to
    # the fixed bits' known value) whenever it applies -- left at False here
    # from the last A/B comparison (see BINARY_ENCODING_FINDINGS.md).
    _FIX_BINARY_START = False

    def _binary_transition_truth_table(self, num_positions, neighbor_codes_fn):
        """
        Shared by both adjacency fits: enumerate the full 2^(2B) truth table
        for one (from, to) bit-pair transition block (B "from" bits followed
        by B "to" bits, bits[b] the coefficient of 2**b per
        paths.bits_to_code) and label each state with its target.

        Target: 0 for a physically adjacent (from, to) pair; for everything
        else, the BFS hop-distance (via neighbor_codes_fn, not bit-Hamming
        distance) from the nearest real neighbor of "from" to "to" — so
        states further from any legal move get a larger target. Codes
        >= num_positions don't decode to any real position (out of the 2^B
        range when num_positions isn't a power of 2); those ghost states
        fall out of this same enumeration for free and get a fixed target
        one worse than the largest real-grid distance.

        Returns (bits_list, targets): bits_list[i] is the i-th 2B-bit tuple
        in itertools.product((0,1), repeat=2B) order, targets[i] its target.
        """
        B = self.B
        n_bits = 2 * B

        dist = self._all_pairs_bfs_distance(num_positions, neighbor_codes_fn)
        max_real_dist = max(dist.values(), default=0)
        ghost_target = max_real_dist + 1

        neighbor_cache = {c: list(neighbor_codes_fn(c)) for c in range(num_positions)}

        def target_for(c_from, c_to):
            if c_from >= num_positions or c_to >= num_positions:
                return ghost_target
            if c_to in neighbor_cache[c_from]:
                return 0
            neighbors = neighbor_cache[c_from]
            if not neighbors:
                return ghost_target
            return min(dist.get((n, c_to), ghost_target) for n in neighbors)

        bits_list = list(itertools.product((0, 1), repeat=n_bits))
        targets = [
            target_for(paths.bits_to_code(bits[:B]), paths.bits_to_code(bits[B:]))
            for bits in bits_list
        ]
        return bits_list, targets

    def _fit_binary_pairwise_block(self, num_positions, neighbor_codes_fn, cap=5.0):
        """Dispatch to the LP or LS adjacency fit per
        _BINARY_ADJACENCY_FIT_METHOD. See _fit_binary_pairwise_block_lp and
        _fit_binary_pairwise_block_ls for what each actually does."""
        if self._BINARY_ADJACENCY_FIT_METHOD == "ls":
            return self._fit_binary_pairwise_block_ls(num_positions, neighbor_codes_fn)
        elif self._BINARY_ADJACENCY_FIT_METHOD == "lp":
            return self._fit_binary_pairwise_block_lp(
                num_positions, neighbor_codes_fn, cap=cap
            )
        raise ValueError(
            f"Unknown _BINARY_ADJACENCY_FIT_METHOD '{self._BINARY_ADJACENCY_FIT_METHOD}', expected 'lp' or 'ls'"
        )

    def _fit_binary_pairwise_block_lp(self, num_positions, neighbor_codes_fn, cap=5.0):
        """
        Max-margin LP fit of a local (2B x 2B) binary QUBO block scoring one
        (position_t, position_t+1) transition. See
        _binary_transition_truth_table for the truth table and target
        definition this fits against.

        The solve: an LP over the *coefficients* of the local block (one
        constraint per truth-table row, not one variable per row) —
        maximize a shared margin m subject to energy(valid) <= 0 and
        energy(invalid) >= m * target. The fit produces all entries of the
        local block jointly (diagonal, within-block, and cross-block), not
        just the cross terms — the conditional from->to structure can only
        be expressed through the cross terms, but the LP is free to use
        diagonal and within-block terms too if that's what realizes it.

        Returns the fitted (2B, 2B) upper-triangular numpy array (row i,
        col j for i<=j; local index b in [0,B) is a "from" bit, B+b is the
        matching "to" bit), or raises RuntimeError if the LP is infeasible
        or finds no positive margin.

        CONFIRMED INFEASIBLE for full grid/graph adjacency at the sizes
        tested (e.g. 5x5, B=5): no quadratic form separates all 25x25
        cell-pair transitions at once, even after excluding the one
        structurally-unfixable all-zero row (whose energy is identically 0
        for *any* coefficient choice, so it's skipped below rather than
        constrained either way). Isolating a single "from" cell's local
        discrimination problem *is* separable — the conflict is global,
        from reusing the same shared diagonal/within-block coefficients
        across all 25 "from" cells while only 25 cross-terms are left to
        carry the rest. Kept for comparison against the LS fit
        (_fit_binary_pairwise_block_ls, the current default — see
        _BINARY_ADJACENCY_FIT_METHOD) rather than deleted.

        O(2^(2B)) truth-table rows — cache the result per builder instance
        (it depends only on B and the static adjacency structure, not on
        robot or timestep) rather than recomputing per (robot, t) pair.
        """
        B = self.B
        success, message, margin, block = self._solve_binary_pairwise_lp_margin(
            num_positions, neighbor_codes_fn, cap=cap
        )
        if not success:
            # True LP infeasibility (res.success=False) — distinct from, and
            # much rarer than, the "solved but margin<=0" case below. Not
            # observed for any grid/graph adjacency case tried so far (the
            # all-zero-coefficients point is always feasible), but scipy's
            # contract doesn't guarantee that in general, so it's still
            # handled.
            raise RuntimeError(
                f"Binary adjacency LP fit infeasible (B={B}, num_positions={num_positions}): {message}"
            )

        margin_tol = 1e-9
        if margin <= margin_tol:
            # linprog reports success whenever it finds the optimum — even
            # when that optimum is m=0, i.e. "no positive margin exists".
            # res.success alone does NOT mean the fit is useful; the margin
            # itself must be checked separately. This is the case that
            # actually fires for full grid/graph adjacency: the LP *is*
            # classifiable (solves cleanly to optimality), it just proves no
            # quadratic form beats the trivial all-zero one.
            raise RuntimeError(
                f"Binary adjacency LP fit found no separating margin "
                f"(m={margin}, B={B}, num_positions={num_positions}) — the "
                f"requested valid/invalid targets aren't realizable by a "
                f"quadratic form over this block."
            )

        return block

    def _solve_binary_pairwise_lp_margin(
        self, num_positions, neighbor_codes_fn, cap=5.0
    ):
        """
        Non-raising core of _fit_binary_pairwise_block_lp: builds and solves
        the same max-margin LP, returning the raw result instead of raising
        on failure. Split out so callers (tests, diagnostics) can tell "LP
        infeasible" (res.success=False) apart from "LP solved but found no
        positive margin" (res.success=True, margin<=0) — both currently
        raise RuntimeError from _fit_binary_pairwise_block_lp, but the
        distinction matters for characterizing *why* a given (B,
        num_positions) case can't be separated.

        Returns (success, message, margin, block): block is always the
        (2B, 2B) array built from res.x (all-zero-ish when infeasible/no
        margin), message is res.message (empty string on success).
        """
        B = self.B
        n_bits = 2 * B
        bits_list, targets = self._binary_transition_truth_table(
            num_positions, neighbor_codes_fn
        )

        pair_idx = list(itertools.combinations(range(n_bits), 2))
        n_diag = n_bits
        n_pair = len(pair_idx)
        m_pos = n_diag + n_pair  # margin variable is the last unknown

        A_ub = []
        b_ub = []
        for bits, target in zip(bits_list, targets):
            if not any(bits):
                # The all-zero bitstring has energy identically 0 for *any*
                # coefficient choice (every term is a product of at least
                # one x_i, so all-zero x kills every term) — no coefficient
                # can move it. Whatever target this row would get is either
                # trivially satisfied (target 0) or structurally
                # unsatisfiable (target > 0, which forces m <= 0 no matter
                # what the rest of the table looks like). Leaving it
                # unconstrained is the honest option: this one specific
                # transition (whichever position decodes to code 0, on both
                # sides) can never be discriminated by a quadratic form, so
                # we don't claim a target for it either way.
                continue

            diag_coeffs = list(bits)
            pair_coeffs = [bits[i] * bits[j] for i, j in pair_idx]

            if target == 0:
                # energy(valid) <= 0
                A_ub.append(diag_coeffs + pair_coeffs + [0])
                b_ub.append(0)
            else:
                # -energy(invalid) + m*target <= 0  <=>  energy >= m*target
                A_ub.append(
                    [-c for c in diag_coeffs] + [-c for c in pair_coeffs] + [target]
                )
                b_ub.append(0)

        c_obj = [0] * (n_diag + n_pair) + [-1]  # maximize m
        bounds = [(-cap, cap)] * (n_diag + n_pair) + [(0, cap)]

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        block = np.zeros((n_bits, n_bits))
        if not res.success:
            return False, res.message, None, block

        margin = res.x[m_pos]
        for i in range(n_diag):
            block[i, i] = res.x[i]
        for (i, j), val in zip(pair_idx, res.x[n_diag:m_pos]):
            block[i, j] = val

        return True, "", margin, block

    def _fit_binary_pairwise_block_ls(self, num_positions, neighbor_codes_fn):
        """
        Least-squares fit of the local (2B x 2B) binary QUBO block: instead
        of requiring a strict separating margin across the entire 2^(2B)
        truth table (confirmed infeasible for full grid/graph adjacency —
        see _fit_binary_pairwise_block_lp), minimize the total squared
        error between energy(x) and target(x) over every row, including the
        all-zero row (harmless here — it just contributes one fixed
        residual, not a feasibility blocker like it is for the LP).

        Always has a solution (via least-norm SVD solve, np.linalg.lstsq)
        and degrades gracefully instead of raising when exact separation
        isn't achievable — appropriate for a soft "reward" bias rather than
        a hard constraint. No coefficient cap: lstsq's minimum-norm solution
        needs no explicit bounding.

        Returns the fitted (2B, 2B) upper-triangular numpy array, same
        layout as _fit_binary_pairwise_block_lp.
        """
        B = self.B
        n_bits = 2 * B
        bits_list, targets = self._binary_transition_truth_table(
            num_positions, neighbor_codes_fn
        )

        pair_idx = list(itertools.combinations(range(n_bits), 2))
        n_diag = n_bits
        n_pair = len(pair_idx)

        A = np.zeros((len(bits_list), n_diag + n_pair))
        for row, bits in enumerate(bits_list):
            A[row, :n_diag] = bits
            for k, (i, j) in enumerate(pair_idx):
                A[row, n_diag + k] = bits[i] * bits[j]
        y = np.array(targets, dtype=float)

        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)

        block = np.zeros((n_bits, n_bits))
        for i in range(n_diag):
            block[i, i] = coeffs[i]
        for (i, j), val in zip(pair_idx, coeffs[n_diag:]):
            block[i, j] = val

        return block

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
        binary = self.encoding == "binary"

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

            if binary:
                # Binary allocates a flat B bits per (robot, timestep)
                # regardless of reachability — no reduction is applied yet
                # (deferred; see get_logical_variables), so unlike one-hot
                # there is no connectivity^step growth to bound.
                vars_at_step = self.B
            elif not flooded:
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
        return sorted(
            active_robots, key=lambda x: self.problem.robots[x].priority, reverse=True
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

            # Clear stale BFS data — solve_qubo_smart will repopulate before next build
            self._active_cells = None
            self.build()

    def reset_problem(self):
        """Reset windowing and restore initial start position if available."""
        for robot_id in self.problem.robots.keys():
            robot = self.problem.robots[robot_id]
            robot.path = []
            robot.current_position = robot.start
            robot.active = True  # Reset active flag when resetting problem
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

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        self.reduction_log = []
        total_fixed = {}
        while True:
            new_fixed = self.diag_fixed_vars()
            if not new_fixed:
                break
            total_fixed.update(new_fixed)
        self.reduction_log = []
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

    def reachable_positions_aggressive(self, robot, start, start_time, end_time):
        """
        Compute reachable positions per time step without backtracking.
        That means once a cell is reached, it won't be revisited in future time steps.

        Args:
            robot: Robot object with current_position and robot_num.
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """
        pass

    # Decided to refactor into multiple functions
    # Easier debugging and previous function had troubles with breaks and variable
    def diag_fixed_vars(self):
        """
        Identify and fix variables based on diagonal coefficients with adjacency validation.

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        if self.encoding == "binary":
            # This diagonal-coefficient heuristic assumes one-hot semantics
            # (one candidate variable == one position). Binary reduction is a
            # separate, not-yet-implemented mechanism (constant/alias bit
            # fixing) — see get_logical_variables().
            return {}
        total_fixed = {}
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

        return total_fixed

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
            )

            if fix_result["needs_bfs_recalc"]:
                # Recalculate reachability and fix all remaining timesteps
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

    def _determine_fixes_for_timestep(
        self,
        vars_at_t,
        should_validate,
        prev_fixed_pos,
        robot_id,
        t,
        type,
        goal,
        adjacency_dict,
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

            result["fixes"][var_idx] = 1
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
                        if not is_valid_move(
                            self.problem, (prev_i, prev_j), (curr_i, curr_j), goal
                        ):
                            self.logger.debug(
                                f"Variable {var_idx} (min coeff) not adjacent, needs BFS recalc"
                            )
                            result["needs_bfs_recalc"] = True
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
    ):
        """
        Recalculate reachable positions and fix variables accordingly.
        Returns the variable index fixed to 1 at curr_t.
        """
        self.logger.debug(
            f"BFS recalculation for robot {robot_id} starting at timestep {curr_t}"
        )

        # Get start position
        prev_i, prev_j, _, _ = paths.decode_position(prev_fixed_pos, self.problem)
        if type == "grid":
            start_pos = (prev_i, prev_j)
        else:
            start_pos = self.problem.graph.get_node_from_position((prev_i, prev_j))

        # Calculate reachable positions for all remaining timesteps in the window
        reachable = self.reachable_positions_aggressive(
            self.problem.robots[robot_id], start_pos, prev_timestep, self.t_max
        )

        self.logger.debug(f"Reachable positions: {reachable}")

        last_fixed_var = None

        # Process each reachable timestep
        for t_check in sorted([k for k in reachable.keys() if k >= curr_t]):
            reachable_positions = reachable[t_check]

            # Get or create variables for reachable positions
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

            # Fix unreachable variables to 0
            if t_check in timestep_vars:
                for var_idx, _ in timestep_vars[t_check]:
                    var_pos = self._get_position_from_var(var_idx, type)
                    if var_pos not in reachable_positions:
                        if var_idx not in total_fixed or total_fixed[var_idx] != 0:
                            total_fixed[var_idx] = 0
                            self.Q, _, log = self.reduce_qubo(
                                {var_idx: 0}, log_reductions=self.log_reductions
                            )
                            self.reduction_log.extend(log)
                            self.logger.debug(f"  Fixed {var_idx} to 0 (unreachable)")

            # Fix one reachable variable to 1
            fixed_var = self._fix_best_reachable_var(reachable_vars, total_fixed)
            if fixed_var and t_check == curr_t:
                last_fixed_var = fixed_var

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

        for pos in reachable_positions:
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

    def _fix_best_reachable_var(self, reachable_vars, total_fixed):
        """Fix the best reachable variable to 1, others to 0. Returns the variable fixed to 1."""
        if not reachable_vars:
            self.logger.minimal("  WARNING: No reachable variables!")
            return None

        if len(reachable_vars) == 1:
            var_idx = reachable_vars[0][0]
            total_fixed[var_idx] = 1
            self.Q, _, log = self.reduce_qubo(
                {var_idx: 1}, log_reductions=self.log_reductions
            )
            self.reduction_log.extend(log)
            self.logger.debug(f"  Fixed {var_idx} to 1 (only reachable)")
            return var_idx

        # Multiple reachable - use min coefficient
        coeffs = [coeff for _, coeff in reachable_vars]
        counts = Counter(coeffs)

        if len(counts) > 1:
            min_coeff = min(counts)
            if counts[min_coeff] == 1:
                fix_dict = {}
                fixed_var = None
                for var_idx, coeff in reachable_vars:
                    if coeff == min_coeff:
                        fix_dict[var_idx] = 1
                        fixed_var = var_idx
                        self.logger.debug(f"  Fixed {var_idx} to 1 (min coeff)")
                    else:
                        fix_dict[var_idx] = 0
                        self.logger.debug(f"  Fixed {var_idx} to 0 (higher coeff)")

                total_fixed.update(fix_dict)
                self.Q, _, log = self.reduce_qubo(
                    fix_dict, log_reductions=self.log_reductions
                )
                self.reduction_log.extend(log)
                return fixed_var

        return None

    def _get_position_from_var(self, var_idx, type):
        """Extract position from variable index."""
        i, j, _, _ = paths.decode_position(var_idx, self.problem)

        if type == "grid":
            return (i, j)
        else:
            return self.problem.graph.get_node_from_position((i, j))
