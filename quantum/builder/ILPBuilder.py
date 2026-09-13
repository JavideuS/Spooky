import pyomo.environ as pyo
from quantum.utils.logger import get_logger
from quantum.pathFormulation import InfeasibleProblemError


def validate_time_horizon(problem):
    """
    Hard-reject a robot whose horizon can't possibly fit a path to goal,
    for exact solvers only (ILP, CBS). They solve the whole horizon in one
    shot -- unlike QUBO's windowed builders, there's no partial/reduced
    result to fall back on and no post-hoc benchmark validity check worth
    running first, so an infeasible horizon here is a hard error, not a
    warning (see BaseILPBuilder/BaseCBSBuilder docstrings: both search
    exactly `robot.T - 1` moves from start via bfs_reachable_sets()).

    Grid-mode only: Manhattan distance is an exact admissible lower bound
    on moves-to-goal there. Graph mode has no equivalent cheap bound
    (edge topology isn't Euclidean), so it's left unchecked here -- same
    reasoning as PathfindingProblem not doing full BFS reachability.
    """
    if problem.grid is None:
        return
    for robot in problem.robots.values():
        dist = problem.manhattan_distance(robot.start, robot.goal)
        if robot.T - 1 < dist:
            raise InfeasibleProblemError(
                f"Robot '{robot.robot_id}' has T={robot.T} (={robot.T - 1} moves) "
                f"but needs at least {dist} (Manhattan distance) to reach its "
                f"goal from its start -- no path can exist in this horizon."
            )


def bfs_reachable_sets(reachable, start, max_steps):
    """Plain (non-aggressive) BFS from start, allowed to stay in place or
    revisit cells each step — reachable[v] must already include v itself.
    Returns a list of sets indexed by step count 0..max_steps: sets[k] is
    every vertex the robot could occupy after exactly k moves. Unlike QUBO's
    non-backtracking reachable_positions_aggressive(), this is exact (a true
    over-approximation is impossible here, and staying/backtracking are both
    legal ILP moves), so it never excludes a feasible cell — it only shrinks
    the ILP's search space, it can't change the optimal solution. The set is
    monotonically non-decreasing and saturates once it covers the robot's
    whole connected component, so this stops growing it early rather than
    recomputing an unchanged set out to max_steps."""
    sets = [{start}]
    for step in range(1, max_steps + 1):
        prev = sets[-1]
        nxt = prev | {n for v in prev for n in reachable[v]}
        sets.append(nxt)
        if len(nxt) == len(prev):
            sets.extend([nxt] * (max_steps - step))
            break
    return sets


def reverse_adjacency(reachable):
    """Build the reverse of a forward adjacency map: reverse[v] lists every u
    with v in reachable[u]. Needed for a goal-anchored backward BFS — reusing
    `reachable` directly for that would silently assume the graph is
    undirected, which grid movement happens to satisfy but isn't guaranteed
    for an arbitrary loaded graph. bfs_reachable_sets(reverse_adjacency(adj),
    goal, k) then gives exactly the set of vertices that can reach `goal`
    within k moves in the original (forward) graph."""
    reverse = {v: [] for v in reachable}
    for u, neighbors in reachable.items():
        for v in neighbors:
            reverse[v].append(u)
    return reverse


class BaseILPBuilder:
    """
    Shared scaffolding for ILP builders: robot-state reset, penalty-set stub
    metadata (ILP has no penalty weights, only hard constraints), and the
    common constructor shape. Concrete grid/graph subclasses implement
    build(), vars_per_time, and local_index(v) — the vertex-representation-
    specific pieces (grid vertices are (i, j) tuples, graph vertices are
    plain node ints). Solves the whole time horizon in one shot — no
    windowing, no Q dict — so this does not inherit from BaseQUBO.
    """

    def __init__(self, problem, name="ilp", verbose_level=2):
        validate_time_horizon(problem)
        self.problem = problem
        self.name = name
        self.model = None
        self.verbose_level = verbose_level
        self.logger = get_logger()
        # No penalty weights — ILP uses hard constraints instead. Named here
        # (fixed by build()'s structure, not config-driven like QUBO's K_*
        # weights) so BenchmarkRunner's `qubobuilder.penalties.get("name", ...)`
        # duck-typing works unchanged and the benchmark JSON still records
        # what's actually enforcing correctness.
        self.penalties = {
            "name": "ilp_hard_constraints",
            "constraints": [
                "one_hot",
                "adjacency",
                "start",
                "goal",
                "goal_lock",
                "crash",
                "swap",
                "trailing",
            ],
        }

    def build(self, preprocess=True):
        """Build the Pyomo ILP model for the current problem state and return it.

        Args:
            preprocess: When True (default), fixes x[a, v, t] to 0 for every
                (v, t) that fails a forward-from-start BFS reachability check
                *or* a backward-from-goal one (see bfs_reachable_sets() /
                reverse_adjacency()) — cells robot a couldn't possibly have
                reached by t, or couldn't possibly still reach its goal from
                by the deadline. Both are exact: any feasible path's own
                prefix/suffix proves its cells pass both checks, so this can
                only shrink the search space HiGHS branches over, never
                exclude a feasible (or optimal) solution. When False, only
                the per-robot active time window is fixed (the minimum
                needed for correctness); the full free-cell set is left open
                at every in-window t.
        """
        raise NotImplementedError

    def local_index(self, v):
        """Map a builder-native vertex (grid tuple or graph node id) to the
        position component of the flat (robot, time, position) variable
        index — the inverse of decode_position()'s per-format unpacking."""
        raise NotImplementedError

    def reset_problem(self):
        """Restore every robot to its initial start state (mirrors BaseQUBO.reset_problem)."""
        for robot in self.problem.robots.values():
            robot.reset()


class GridILPBuilder(BaseILPBuilder):
    """ILP builder for grid pathfinding problems."""

    def __init__(self, problem, name="ilp_grid", verbose_level=2):
        if problem.grid is None:
            raise ValueError("Grid representation not available in this problem")
        super().__init__(problem, name=name, verbose_level=verbose_level)
        self.vars_per_time = problem.grid.M * problem.grid.N

    def local_index(self, v):
        i, j = v
        return i * self.problem.grid.N + j

    def build(self, preprocess=True):
        problem = self.problem
        grid = problem.grid
        robots = problem.robots

        model = pyo.ConcreteModel()

        model.A = pyo.Set(initialize=list(robots.keys()))
        model.V = pyo.Set(
            initialize=[
                (i, j)
                for i in range(grid.M)
                for j in range(grid.N)
                if (i, j) not in grid.obstacles
            ]
        )
        model.T = pyo.RangeSet(0, problem.T - 1)
        model.T_minus = pyo.RangeSet(0, problem.T - 2)

        # ILP allows staying in place; QUBO's grid.adjacency is kept strict-neighbor-only
        # since it relies on that for its own penalty terms, so extend it locally here.
        reachable = {v: grid.adjacency[v] + [v] for v in model.V}

        # Each robot only exists for its own [start_time, start_time + T - 1] window,
        # matching the QUBO: no variables before start_time, none after the robot's
        # own goal is reached (regardless of how long the shared horizon runs).
        active_range = {
            a: range(robots[a].start_time, robots[a].start_time + robots[a].T)
            for a in robots
        }

        # Forward-from-start and backward-from-goal BFS reachability per robot:
        # cells it cannot possibly have reached by t, or cannot possibly still
        # reach goal from by the deadline, are fixed to 0. See
        # bfs_reachable_sets() / reverse_adjacency() docstrings for why the
        # intersection of both is exact.
        goal_deadline = {a: robots[a].start_time + robots[a].T - 1 for a in robots}
        if preprocess:
            reverse_reachable = reverse_adjacency(reachable)
            forward_sets = {
                a: bfs_reachable_sets(
                    reachable, robots[a].current_position, robots[a].T - 1
                )
                for a in robots
            }
            backward_sets = {
                a: bfs_reachable_sets(
                    reverse_reachable, robots[a].goal, robots[a].T - 1
                )
                for a in robots
            }
        else:
            forward_sets = backward_sets = None
        self.logger.debug(
            f"Forward sets: {forward_sets}\nBackward sets: {backward_sets}"
        )

        # Clearance (quantum/utils/clearance.py). Both are empty / all-{(0,0)}
        # at the defaults, so everything below is a no-op and the model becomes
        # identical to the pre-clearance one.
        keepout = problem.get_obstacle_keepout()  # {a: frozenset[(i,j)]}
        clearance_table = problem.get_clearance_table()  # {(a,b): D_ab}
        _trivial = frozenset({(0, 0)})
        robot_list = list(robots.keys())
        # Clearance robot pairs: those whose clearance shell (D_ab) is
        # bigger than one cell.
        clearance_robot_pairs = [
            (robot_list[i], robot_list[j], D)
            for i in range(len(robot_list))
            for j in range(i + 1, len(robot_list))
            if (D := clearance_table.get((robot_list[i], robot_list[j])))
            and D != _trivial
        ]
        clearance_robot_pair_ids = {(a, b) for a, b, _ in clearance_robot_pairs}

        # Decision variables x[robot, position, time]
        model.x = pyo.Var(model.A, model.V, model.T, within=pyo.Binary)

        # legal[a][t] = cells robot a can still be a leader on at t (survives the
        # window / BFS-reachability / obstacle keep-out fixing below). Reused to
        # prune the clearance constraints.
        # A robot whose start already is its goal is pinned there for its whole
        # window (start + goal + goal_lock + one_hot leave it no freedom), so
        # fix every other cell to 0 outright and keep its `legal` set to a
        # single cell so the clearance enumeration below doesn't range it over the map.
        parked = {a for a in model.A if robots[a].current_position == robots[a].goal}

        legal = {a: {} for a in model.A}
        in_window_vars = 0
        bfs_fixed = 0
        keepout_fixed = 0
        parked_fixed = 0
        for a in model.A:
            ko_a = keepout.get(a, frozenset())
            for t in model.T:
                if t not in active_range[a]:
                    for v in model.V:
                        model.x[a, v, t].fix(0)
                    continue
                if a in parked:
                    goal = robots[a].goal
                    for v in model.V:
                        in_window_vars += 1
                        if v != goal:
                            model.x[a, v, t].fix(0)
                            parked_fixed += 1
                    legal[a][t] = {goal}
                    continue
                legal_at = set()
                for v in model.V:
                    in_window_vars += 1
                    if v in ko_a:
                        model.x[a, v, t].fix(0)
                        keepout_fixed += 1
                        continue
                    if preprocess:
                        forward_ok = v in forward_sets[a][t - robots[a].start_time]
                        backward_ok = v in backward_sets[a][goal_deadline[a] - t]
                        if not (forward_ok and backward_ok):
                            model.x[a, v, t].fix(0)
                            bfs_fixed += 1
                            continue
                    legal_at.add(v)
                legal[a][t] = legal_at
        reduced = bfs_fixed + keepout_fixed + parked_fixed
        self.bfs_stats = {
            "window": 0,
            "preprocess": preprocess,
            "initial_variables": in_window_vars,
            "variables_reduced": reduced,
            "obstacle_keepout_fixed": keepout_fixed,
            "parked_fixed": parked_fixed,
            "final_variables": in_window_vars - reduced,
            "reduction_ratio": round(reduced / in_window_vars, 4)
            if in_window_vars
            else 0,
        }

        # exactly one cell per robot per timestep, only while the robot exists
        def one_hot_rule(m, a, t):
            if t not in active_range[a]:
                return pyo.Constraint.Skip
            return sum(m.x[a, v, t] for v in m.V) == 1

        model.one_hot = pyo.Constraint(model.A, model.T, rule=one_hot_rule)

        # if at v at time t, must move to a neighbor of v at time t+1, only while
        # both t and t+1 fall inside the robot's own window
        def adjacency_rule(m, a, i, j, t):
            if t not in active_range[a] or (t + 1) not in active_range[a]:
                return pyo.Constraint.Skip
            return m.x[a, (i, j), t] <= sum(
                m.x[a, vp, t + 1] for vp in reachable[(i, j)]
            )

        model.adjacency = pyo.Constraint(
            model.A, model.V, model.T_minus, rule=adjacency_rule
        )

        # at start position at each robot's own start time
        model.start = pyo.Constraint(
            model.A,
            rule=lambda m, a: (
                m.x[a, robots[a].current_position, robots[a].start_time] == 1
            ),
        )

        # at goal position at each robot's own goal time
        model.goal = pyo.Constraint(
            model.A,
            rule=lambda m, a: (
                m.x[a, robots[a].goal, robots[a].start_time + robots[a].T - 1] == 1
            ),
        )

        # once at goal, stay at goal, only within the robot's own window
        def goal_lock_rule(m, a, t):
            if t not in active_range[a] or (t + 1) not in active_range[a]:
                return pyo.Constraint.Skip
            return m.x[a, robots[a].goal, t] <= m.x[a, robots[a].goal, t + 1]

        model.goal_lock = pyo.Constraint(model.A, model.T_minus, rule=goal_lock_rule)

        # at most one robot per cell per timestep. Also the (0,0) case of every
        # clearance robot pair's separation, so crash_clearance below only adds the rest of
        # the D_ab shell.
        model.crash = pyo.Constraint(
            model.V,
            model.T,
            rule=lambda m, i, j, t: sum(m.x[a, (i, j), t] for a in m.A) <= 1,
        )

        # No two robots may swap positions across an edge between t and t+1.
        # Skipped for clearance robot pairs not because swap stops applying,
        # but because model.trailing below provably dominates it), so every
        # case this constraint would forbid, trailing already forbids too.
        model.swap = pyo.ConstraintList()
        for t in model.T_minus:
            for idx_i in range(len(robot_list)):
                for idx_j in range(idx_i + 1, len(robot_list)):
                    ai, aj = robot_list[idx_i], robot_list[idx_j]
                    if (ai, aj) in clearance_robot_pair_ids:
                        continue
                    for v in model.V:
                        for w in grid.adjacency[v]:
                            model.swap.add(
                                model.x[ai, v, t]
                                + model.x[aj, w, t]
                                + model.x[ai, w, t + 1]
                                + model.x[aj, v, t + 1]
                                <= 3
                            )

        self._add_clearance_constraints(
            model, active_range, legal, clearance_robot_pairs
        )

        # minimize total timesteps spent away from goal (equivalent to sum of arrival times,
        # given goal_lock forces x[a, g_a, ·] to be monotone)
        model.obj = pyo.Objective(
            sense=pyo.minimize,
            expr=sum(
                model.x[a, v, t]
                for a in model.A
                for t in model.T
                for v in model.V
                if v != robots[a].goal
            ),
        )

        extra = ""
        if clearance_robot_pairs:
            extra = (
                f", clearance: {len(clearance_robot_pairs)} clearance robot pair(s), "
                f"{len(model.crash_clearance)} crash + "
                f"{len(model.trailing)} trailing rows"
            )
        self.logger.standard(
            f"ILP model built: {len(model.A)} robots, {len(model.V)} free cells, "
            f"{problem.T} timesteps{extra}"
            f"\nBFS Stats: {self.bfs_stats}"
        )
        self.model = model
        return self.model

    def _add_clearance_constraints(
        self, model, active_range, legal, clearance_robot_pairs
    ):
        """Two DIFFERENT constraints for clearance robot pairs (clearance shell
        bigger than one cell), not one generalised in two places.

          crash_clearance, at (ta,tb,offsets) = (t, t, D\\{(0,0)}):
              the rest of the SAME-time footprint-overlap shell.
              It is model.crash generalised from a point to a disc.
              The (0,0) centre is already covered by the compact global model.crash.

          trailing, at (t+1, t, D) and (t, t+1, D):
              a NEW hazard with no point-robot analogue and no relation to
              swap: does one robot's endpoint at one end of the step land in
              the other's shell at the other end of the step? Point robots
              have instantaneous, exact occupancy, so there is no such thing
              as "lingering" in a cell partway through vacating it,
              this guards against real, continuous, imperfectly-synchronised
              robot motion, where that lingering is physically real. Uses
              the full D (0,0 included): "a arrives where b just was" counts
              regardless of whether b has since moved on, which is why this
              is not just swap-with-a-wider-D.

        Each disjunct only involves two position variables (one robot's
        endpoint at one time, the other's at the other time) -- not all
        four -- so no transition/adjacency reasoning is needed here at all:
        banning the pair directly is an exact, tighter encoding of the same
        predicate than enumerating full (from, to) transition tuples would
        be. No-op when clearance_robot_pairs is empty."""
        model.crash_clearance = pyo.ConstraintList()
        model.trailing = pyo.ConstraintList()
        if not clearance_robot_pairs:
            return

        def add_shell_rows(target, a, ta, b, tb, offsets):
            la = legal[a].get(ta)
            lb = legal[b].get(tb)
            if not la or not lb:
                return
            for u in la:
                near = [w for d in offsets if (w := (u[0] - d[0], u[1] - d[1])) in lb]
                if near:
                    target.add(
                        model.x[a, u, ta] + sum(model.x[b, w, tb] for w in near) <= 1
                    )

        for a, b, D in clearance_robot_pairs:
            off_nz = [d for d in D if d != (0, 0)]
            shared_t = set(active_range[a]) & set(active_range[b])
            for t in shared_t:
                add_shell_rows(model.crash_clearance, a, t, b, t, off_nz)
                if (t + 1) in active_range[a] and (t + 1) in active_range[b]:
                    add_shell_rows(model.trailing, a, t + 1, b, t, D)
                    add_shell_rows(model.trailing, a, t, b, t + 1, D)


class GraphILPBuilder(BaseILPBuilder):
    """ILP builder for graph pathfinding problems. Same constraint structure
    as GridILPBuilder, written over the graph's native vertex set instead of
    (i, j) grid cells — see BaseILPBuilder's docstring for what differs."""

    def __init__(self, problem, name="ilp_graph", verbose_level=2):
        if problem.graph is None:
            raise ValueError("Graph representation not available in this problem")
        super().__init__(problem, name=name, verbose_level=verbose_level)
        self.vars_per_time = len(problem.graph.nodes)

    def local_index(self, v):
        return v

    def build(self, preprocess=True):
        problem = self.problem
        graph = problem.graph
        robots = problem.robots

        model = pyo.ConcreteModel()

        model.A = pyo.Set(initialize=list(robots.keys()))
        model.V = pyo.Set(initialize=list(range(len(graph.nodes))))
        model.T = pyo.RangeSet(0, problem.T - 1)
        model.T_minus = pyo.RangeSet(0, problem.T - 2)

        # graph.adjacency[v] is a set of (neighbor_id, weight) pairs — drop the
        # weight and add the self-loop, same "stay in place" allowance grid gets.
        reachable = {v: [n for n, _w in graph.adjacency[v]] + [v] for v in model.V}

        # Robot start/goal may be stored as raw node ids or as coordinates,
        # depending on how the problem was built — resolve both to node ids
        # once, up front, via the same helper the rest of the codebase uses
        # for graph robot state (see PathfindingProblem.get_graph_robot_current_goal).
        start_goal = {a: problem.get_graph_robot_current_goal(a) for a in robots}

        active_range = {
            a: range(robots[a].start_time, robots[a].start_time + robots[a].T)
            for a in robots
        }

        # Forward-from-start and backward-from-goal BFS reachability per robot:
        # nodes it cannot possibly have reached by t, or cannot possibly still
        # reach goal from by the deadline, are fixed to 0. See
        # bfs_reachable_sets() / reverse_adjacency() docstrings for why the
        # intersection of both is exact.
        goal_deadline = {a: robots[a].start_time + robots[a].T - 1 for a in robots}
        if preprocess:
            reverse_reachable = reverse_adjacency(reachable)
            forward_sets = {
                a: bfs_reachable_sets(reachable, start_goal[a][0], robots[a].T - 1)
                for a in robots
            }
            backward_sets = {
                a: bfs_reachable_sets(
                    reverse_reachable, start_goal[a][1], robots[a].T - 1
                )
                for a in robots
            }
        else:
            forward_sets = backward_sets = None
        self.logger.debug(
            f"Forward sets: {forward_sets}\nBackward sets: {backward_sets}"
        )
        # Decision variables x[robot, node, time]
        model.x = pyo.Var(model.A, model.V, model.T, within=pyo.Binary)

        in_window_vars = 0
        bfs_fixed = 0
        for a in model.A:
            for v in model.V:
                for t in model.T:
                    if t not in active_range[a]:
                        model.x[a, v, t].fix(0)
                        continue
                    in_window_vars += 1
                    if preprocess:
                        forward_ok = v in forward_sets[a][t - robots[a].start_time]
                        backward_ok = v in backward_sets[a][goal_deadline[a] - t]
                        if not (forward_ok and backward_ok):
                            model.x[a, v, t].fix(0)
                            bfs_fixed += 1
        self.bfs_stats = {
            "window": 0,
            "preprocess": preprocess,
            "initial_variables": in_window_vars,
            "variables_reduced": bfs_fixed,
            "final_variables": in_window_vars - bfs_fixed,
            "reduction_ratio": round(bfs_fixed / in_window_vars, 4)
            if in_window_vars
            else 0,
        }

        # exactly one node per robot per timestep, only while the robot exists
        def one_hot_rule(m, a, t):
            if t not in active_range[a]:
                return pyo.Constraint.Skip
            return sum(m.x[a, v, t] for v in m.V) == 1

        model.one_hot = pyo.Constraint(model.A, model.T, rule=one_hot_rule)

        # if at v at time t, must move to a neighbor of v at time t+1, only while
        # both t and t+1 fall inside the robot's own window
        def adjacency_rule(m, a, v, t):
            if t not in active_range[a] or (t + 1) not in active_range[a]:
                return pyo.Constraint.Skip
            return m.x[a, v, t] <= sum(m.x[a, vp, t + 1] for vp in reachable[v])

        model.adjacency = pyo.Constraint(
            model.A, model.V, model.T_minus, rule=adjacency_rule
        )

        # at start node at each robot's own start time
        model.start = pyo.Constraint(
            model.A,
            rule=lambda m, a: m.x[a, start_goal[a][0], robots[a].start_time] == 1,
        )

        # at goal node at each robot's own goal time
        model.goal = pyo.Constraint(
            model.A,
            rule=lambda m, a: (
                m.x[a, start_goal[a][1], robots[a].start_time + robots[a].T - 1] == 1
            ),
        )

        # once at goal, stay at goal, only within the robot's own window
        def goal_lock_rule(m, a, t):
            if t not in active_range[a] or (t + 1) not in active_range[a]:
                return pyo.Constraint.Skip
            goal_node = start_goal[a][1]
            return m.x[a, goal_node, t] <= m.x[a, goal_node, t + 1]

        model.goal_lock = pyo.Constraint(model.A, model.T_minus, rule=goal_lock_rule)

        # at most one robot per node per timestep
        model.crash = pyo.Constraint(
            model.V, model.T, rule=lambda m, v, t: sum(m.x[a, v, t] for a in m.A) <= 1
        )

        # no two robots may swap positions across an edge between t and t+1
        model.swap = pyo.ConstraintList()
        robot_list = list(robots.keys())
        for t in model.T_minus:
            for idx_i in range(len(robot_list)):
                for idx_j in range(idx_i + 1, len(robot_list)):
                    ai, aj = robot_list[idx_i], robot_list[idx_j]
                    for v in model.V:
                        for w, _weight in graph.adjacency[v]:
                            model.swap.add(
                                model.x[ai, v, t]
                                + model.x[aj, w, t]
                                + model.x[ai, w, t + 1]
                                + model.x[aj, v, t + 1]
                                <= 3
                            )

        # minimize total timesteps spent away from goal (equivalent to sum of arrival times,
        # given goal_lock forces x[a, g_a, ·] to be monotone)
        model.obj = pyo.Objective(
            sense=pyo.minimize,
            expr=sum(
                model.x[a, v, t]
                for a in model.A
                for t in model.T
                for v in model.V
                if v != start_goal[a][1]
            ),
        )

        self.logger.standard(
            f"ILP model built: {len(model.A)} robots, {len(model.V)} nodes, "
            f"{problem.T} timesteps"
            f"\nBFS Stats: {self.bfs_stats}"
        )
        self.model = model
        return self.model
