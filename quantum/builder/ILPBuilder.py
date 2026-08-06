import pyomo.environ as pyo
from quantum.utils.logger import get_logger


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
                a: bfs_reachable_sets(reverse_reachable, robots[a].goal, robots[a].T - 1)
                for a in robots
            }
        else:
            forward_sets = backward_sets = None
        self.logger.debug(f"Forward sets: {forward_sets}\nBackward sets: {backward_sets}")
        # Decision variables x[robot, position, time]
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

        # at most one robot per cell per timestep
        model.crash = pyo.Constraint(
            model.V,
            model.T,
            rule=lambda m, i, j, t: sum(m.x[a, (i, j), t] for a in m.A) <= 1,
        )

        # no two robots may swap positions across an edge between t and t+1
        model.swap = pyo.ConstraintList()
        robot_list = list(robots.keys())
        for t in model.T_minus:
            for idx_i in range(len(robot_list)):
                for idx_j in range(idx_i + 1, len(robot_list)):
                    ai, aj = robot_list[idx_i], robot_list[idx_j]
                    for v in model.V:
                        for w in grid.adjacency[v]:
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
                if v != robots[a].goal
            ),
        )

        self.logger.standard(
            f"ILP model built: {len(model.A)} robots, {len(model.V)} free cells, "
            f"{problem.T} timesteps"
            f"\nBFS Stats: {self.bfs_stats}"
        )
        self.model = model
        return self.model


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
        self.logger.debug(f"Forward sets: {forward_sets}\nBackward sets: {backward_sets}")
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
