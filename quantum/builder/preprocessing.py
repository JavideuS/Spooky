"""
Pre-processing of a QUBO window, shared by the grid and graph builders:

- reachability: which positions a robot may occupy at each window step
  (reachable_aggressive / reachable_safe). The builders pass their own
  neighbours, so the two representations can't drift apart -- a separate
  graph copy once added (node, weight) pairs to the set as if they were
  nodes, and every bfs_safe graph window crashed.
- the diagonal fixer (DiagonalFixer): pins variables from their own
  coefficients, robot by robot, step by step -- greedy prioritized planning
  that keeps each robot's pinned path connected.

The pre-processing modes that combine the two live in quantum.utils.preprocess.
"""

from collections import Counter

from quantum.utils import paths
from quantum.utils.validation import is_valid_move


def reachable_aggressive(
    start, goal, neighbours, start_time, end_time, blocked=None, allow_wait_at=None
):
    """
    Positions reachable at each step without revisiting: once a position is
    reached it is never offered again, except the goal, which stays available
    once reached (so the goal lock never conflicts with pruning).

    Args:
        start: position at start_time.
        goal: the robot's goal position.
        neighbours: position -> iterable of adjacent positions.
        start_time, end_time: window steps, end exclusive.
        blocked: optional {t: {position, ...}} to treat as obstacles at that
            step only -- used when recalculating around a detected collision,
            so the colliding position can't be handed back as "the only
            option" again.
        allow_wait_at: optional set of steps at which, if blocking and "no
            revisit" leave nothing to expand to, the robot may stay where it
            is (minus blocked positions) instead of the walk ending early.

    Returns:
        {t: {position, ...}} from start_time; stops early when nothing is
        reachable.
    """
    blocked = blocked or {}
    allow_wait_at = allow_wait_at or set()
    reachable = {start_time: {start}}
    visited = {start}
    for t in range(start_time + 1, end_time):
        prev_layer = reachable[t - 1]
        curr_layer = set()
        blocked_at_t = blocked.get(t, set())
        for pos in prev_layer:
            for nxt in neighbours(pos):
                if nxt not in visited and nxt not in blocked_at_t:
                    curr_layer.add(nxt)
                    visited.add(nxt)
        if goal in visited and goal not in blocked_at_t:
            curr_layer.add(goal)
        if not curr_layer:
            # The "stay" option must still respect blocked_at_t, or a position
            # blocked for being a confirmed collision is re-offered on every retry
            stay_layer = set(prev_layer) - blocked_at_t if t in allow_wait_at else set()
            if not stay_layer:
                break
            curr_layer = stay_layer
        reachable[t] = curr_layer
    return reachable


def reachable_safe(start, neighbours, start_time, end_time, blocked=None):
    """
    Monotone reachability: the set at t is the set at t-1 plus its
    neighbours, so staying put and revisiting are both allowed. Same
    semantics as ILPBuilder.bfs_reachable_sets(), and the only variant that
    can't exclude a feasible solution -- reachable_aggressive() forbids
    waiting, which is how robots yield. Stops growing once it saturates.
    `blocked` ({t: {position, ...}}) is removed from the set at t (cells a
    parked robot holds, BaseQUBO.parked_cells).

    Returns:
        {t: {position, ...}} for every step in [start_time, end_time).
    """
    blocked = blocked or {}
    current = {start}
    reachable = {start_time: set(current)}
    for t in range(start_time + 1, end_time):
        grown = current | {n for pos in current for n in neighbours(pos)}
        grown -= blocked.get(t, set())
        if not blocked and len(grown) == len(current):
            for rest in range(t, end_time):
                reachable[rest] = set(current)
            break
        current = grown
        reachable[t] = set(current)
    return reachable


class DiagonalFixer:
    """
    The diagonal fixer, as a mixin of BaseQUBO (it reads and reduces the
    builder's Q and calls its reachability and reduction methods).
    reduce_diag_fixed_vars_iterative() is the entry point.
    """

    def reduce_diag_fixed_vars_iterative(self, prior_fixed=None):
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

        Args:
            prior_fixed: {var_idx: 1} pinned before this pass, i.e.
                get_logical_variables()'s BFS pins (starts, parked or
                instantaneous-goal robots). Only _rival_keepout reads them,
                so a robot routes around a rival BFS already committed; they
                are kept out of total_fixed so recalculation never unfixes
                or re-folds them.

        Returns:
            dict: {variable_index: fixed_value} where fixed_value is 0 or 1
        """
        self.reduction_log = []
        self._pinned_before_diag = dict(prior_fixed or {})
        total_fixed = {}
        while True:
            new_fixed = self.diag_fixed_vars(prior_fixed=total_fixed)
            if not new_fixed:
                break
            total_fixed.update(new_fixed)
        self.reduction_log = []  # Memory cleanup
        self._pinned_before_diag = {}
        return total_fixed

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
                # should_validate only checks a 1-pin against the step right
                # before it. Pins at other steps -- from this pass or an
                # earlier one -- can still cut the robot's chain: robot at A,
                # A+2 pinned 1 at t+2 in one pass, then A+1 (the only way
                # between) pinned 0 at t+1 in the next. Leave the step to the
                # solver if its pins would disconnect any two consecutive
                # 1-pins of this robot.
                if fix_result["fixes"] and not self._chain_connected(
                    robot_offset, vars_per_time, total_fixed, fix_result["fixes"]
                ):
                    fix_result["fixes"] = {}
                # fixer="prefix": commit a path only as an unbroken run of
                # 1-pins -- a step without one (a tie for the lowest
                # coefficient) ends this robot's pinning for the pass; the
                # next pass resumes once other robots are fixed. Pins less,
                # which suits SA (more full_safe valid) but costs QAOA qubits.
                if self.fixer == "prefix" and not any(fix_result["fixes"].values()):
                    break
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

    def _chain_connected(self, robot_offset, vars_per_time, total_fixed, new_fixes):
        """Whether, with new_fixes applied on top of total_fixed, every pair of
        consecutive 1-pins of this robot in the window (its BFS start pin
        included) is still connected through open cells: at each step in
        between, a cell is open if it is still in Q or pinned to 1 and not
        pinned to 0. Moves are to a neighbour or staying put. Grid only -- the
        graph builder keeps the previous (unchecked) behaviour."""
        if self.problem.get_format_type() != "grid":
            return True
        N = self.problem.grid.N
        adjacency = self.problem.grid.adjacency
        span = vars_per_time * self.total_t
        pinned = {
            **getattr(self, "_pinned_before_diag", {}),
            **total_fixed,
            **new_fixes,
        }

        def mine(var):
            return robot_offset <= var < robot_offset + span

        ones = sorted(
            divmod(var - robot_offset, vars_per_time)
            for var, val in pinned.items()
            if val == 1 and mine(var)
        )

        def is_open(t, cell):
            var = robot_offset + t * vars_per_time + cell[0] * N + cell[1]
            if var in pinned:
                return pinned[var] == 1
            return (var, var) in self.Q

        for (t0, c0), (t1, c1) in zip(ones, ones[1:]):
            if t1 == t0:
                continue
            frontier = {divmod(c0, N)}
            for t in range(t0 + 1, t1 + 1):
                frontier = {
                    d
                    for c in frontier
                    for d in (c, *adjacency.get(c, ()))
                    if is_open(t, d)
                }
                if not frontier:
                    return False
            if divmod(c1, N) not in frontier:
                return False
        return True

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

    def _rival_keepout(self, t, total_fixed, own_robot_num):
        """
        Cells the robot currently being processed may not occupy at
        window-relative timestep `t`, given every *different* robot already
        committed to 1 in `total_fixed`. Mirrors is_solution_valid()'s checks
        (quantum/benchmark/benchmark.py) so a fix this allows is one the
        validator accepts:
        - trivial pair (D_ab == {(0,0)}): the rival's exact cell at t only.
          Moving into a cell the rival just vacated stays legal, as in the
          validator.
        - clearance pair: the D_ab shell around the rival at t
          ("clearance_conflict"), plus the shells around it at t-1 and t+1
          ("trailing_conflict": own cell at t vs rival's at the other end of
          either step).

        Rivals are read from total_fixed plus the BFS pins handed to
        reduce_diag_fixed_vars_iterative(), so a parked robot BFS pinned is
        seen too. Robots are processed one at a time in priority order
        (get_active_robot_in_window()), each fully fixed before the next
        starts, so any rival entry in total_fixed at this point is final for
        this pass -- there's nothing to arbitrate here, just something to
        avoid walking into. The keep-out is also what BFS recalculation
        blocks, so a whole shell is routed around at once instead of one
        cell per retry.

        Robots parked on their goal (no variables from their arrival or
        deadline on) are kept out through BaseQUBO.parked_cells() instead.
        """
        keepout = set()
        committed = {**getattr(self, "_pinned_before_diag", {}), **(total_fixed or {})}
        if not committed or own_robot_num is None:
            return keepout
        clearance_table = self.problem.get_clearance_table()
        num_to_id = {num: rid for rid, num in self.problem.get_robot_nums().items()}
        own_id = num_to_id.get(own_robot_num)
        trivial = frozenset({(0, 0)})
        for fixed_idx, val in committed.items():
            if val != 1:
                continue
            fi, fj, ft, frn = paths.decode_position(fixed_idx, self.problem)
            if frn == own_robot_num or abs(ft - t) > 1:
                continue
            D = clearance_table.get((own_id, num_to_id.get(frn)))
            if not D or D == trivial:
                if ft == t:
                    keepout.add((fi, fj))
                continue
            # (own - rival) in D_ab  <=>  own in rival + D_ab
            keepout.update((fi + di, fj + dj) for di, dj in D)
        return keepout

    def _collides_with_fixed_rival(self, var_idx, t, total_fixed, own_robot_num):
        """
        True if `var_idx` (a candidate cell for the robot currently being
        processed, at window-relative timestep `t`) falls in the keep-out of
        a *different* robot already committed to 1 -- its exact cell, or for
        a clearance pair its D_ab shell at t-1/t/t+1 (see _rival_keepout).
        """
        i, j, _, _ = paths.decode_position(var_idx, self.problem)
        return (i, j) in self._rival_keepout(t, total_fixed, own_robot_num)

    def _collides_with_swap_rival(
        self, var_idx, t, prev_fixed_pos, total_fixed, own_robot_num
    ):
        """
        True if picking `var_idx` at window-relative `t` would swap places
        with a different, already-committed robot: that rival occupied
        var_idx's cell at t-1, and is *also* fixed to move into this robot's
        own t-1 cell at t -- both crossing through each other's edge at the
        same transition. _collides_with_fixed_rival only catches same-
        cell/same-time (vertex) collisions; this is the other half K_trail is
        supposed to cover but never gets a say in, same reason as the vertex
        case (see _collides_with_fixed_rival's docstring).

        Exact-cell only. That is all a trivial pair needs; for a clearance
        pair a swap always lands the candidate in the rival's t-1 shell, so
        _rival_keepout's trailing check already rejects it.
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
                if frn2 == frn and ft2 == t and (fi2, fj2) == (own_prev_i, own_prev_j):
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
        for t_block, cells in self.parked_cells(robot_id).items():
            blocked.setdefault(t_block, set()).update(cells)
        wait_at = set()
        own_robot_num = self.problem.get_robot_nums()[robot_id]
        last_fixed_var = None

        # Route around every committed rival's keep-out up front -- for a
        # clearance pair that's a whole D_ab shell per timestep, which
        # discovering one cell per retry would exhaust the retry budget on.
        for t_block in range(curr_t, self.t_max):
            keepout = self._rival_keepout(t_block, total_fixed, own_robot_num)
            if keepout:
                blocked.setdefault(t_block, set()).update(keepout)

        for _attempt in range(2 * max(self.t_max, 1)):
            reachable = self.reachable_positions_aggressive(
                self.problem.robots[robot_id],
                start_pos,
                prev_timestep,
                self.t_max,
                blocked=blocked,
                allow_wait_at=wait_at,
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
