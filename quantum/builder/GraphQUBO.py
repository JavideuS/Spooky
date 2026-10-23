from collections import deque

from .base_qubo import BaseQUBO
from .preprocessing import reachable_aggressive, reachable_safe

GOAL_OBJECTIVES = ("shaped", "terminal")


class GraphQUBO(BaseQUBO):
    def __init__(
        self,
        problem,
        penalties,
        name="graph",
        var_limit=156,  # 65 131
        window_max_steps=None,
        window_sizing="estimate",
        robot_window_limits=None,
        log_reductions=True,
        verbose_level=2,
        goal_objective="shaped",
        goal_weight=0.5,
        fixer="chain",
    ):
        """
        goal_objective / goal_weight: the arrival-time goal term, as in
        GridQUBOBuilder -- w per step not at the goal plus the distance left
        in steps (BFS over the graph's edges, one edge per timestep; edge
        weights only scale the adjacency reward).
        """
        if goal_objective not in GOAL_OBJECTIVES:
            raise ValueError(
                f"goal_objective must be one of {GOAL_OBJECTIVES}, got {goal_objective!r}"
            )
        if float(goal_weight) <= 0:
            raise ValueError(f"goal_weight must be > 0, got {goal_weight!r}")
        self.goal_objective = goal_objective
        self.goal_weight = float(goal_weight)
        self._goal_distance_maps = {}
        self.graph = problem.graph
        self.num_nodes = len(self.graph.nodes)
        super().__init__(
            problem,
            penalties,
            name=name,
            var_limit=var_limit,
            window_max_steps=window_max_steps,
            window_sizing=window_sizing,
            fixer=fixer,
            robot_window_limits=robot_window_limits,
            log_reductions=log_reductions,
            verbose_level=verbose_level,
        )
        # Multi-robot support: calculate total variables for all robots
        # Use problem.T (total timeline) not total_t (window size) to match QUBOBuilder
        self.initial_num_vars = (
            self.num_nodes * self.problem.T * self.problem.num_robots
        )

        self.logger.standard("Window max steps:", self.max_window_size())

    def build(self, constraints_to_apply=None):
        """Build the QUBO dictionary for graph-based pathfinding."""
        self._warn_if_unrestricted_build(self.num_nodes)
        if constraints_to_apply is None:
            penalty_to_constraint = {
                "K_hot": "one_hot",
                "K_adj": "adjacency_reward",
                "K_start": "start",
                "K_goal": "goal_objective",
                "K_lock": "lock",
                "K_crash": "crash",
                "K_trail": "trailing",
            }
            constraints_to_apply = [
                v for k, v in penalty_to_constraint.items() if k in self.penalties
            ]
            # apply_trailing_penalty already applies the same-node/same-time term
            # itself (weighted by K_crash), so don't also run apply_crash_penalty
            # separately — that would double-count it.
            if "trailing" in constraints_to_apply and "crash" in constraints_to_apply:
                constraints_to_apply.remove("crash")

        self.Q = {}

        if "one_hot" in constraints_to_apply:
            self.apply_one_hot()
        if "start" in constraints_to_apply:
            self.apply_start_penalty()
        if "goal_objective" in constraints_to_apply:
            self.apply_goal_objective()
        if "lock" in constraints_to_apply:
            self.apply_lock_after_goal()
        if "adjacency_reward" in constraints_to_apply:
            self.apply_adjacency_reward()
        if "crash" in constraints_to_apply:
            self.apply_crash_penalty()
        if "trailing" in constraints_to_apply:
            self.apply_trailing_penalty()

        return self.Q

    def _nodes(self, robot_id, t):
        """Return active node IDs for (robot_id, t), falling back to all nodes."""
        if self._active_cells is not None:
            return self._active_cells.get((robot_id, t), [])
        return range(self.num_nodes)

    def get_logical_variables(self, bfs_variant=None):
        """
        Returns (fixed_ones, active_cells):
        - fixed_ones: {flat_idx: 1} for variables known to be 1.
          Absent entries are implicitly 0 — no zeros stored.
        - active_cells: {(robot_id, t): [node_id, ...]} built directly from BFS.
        """
        fixed_ones = {}
        active_cells = {}
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            robot = self.problem.robots[robot_id]

            start, end = self.window_steps(robot_id)

            start_node, goal_node = self.problem.get_graph_robot_current_goal(robot_id)
            start_idx = start_node + (self.num_nodes * start) + robot_offset
            fixed_ones[start_idx] = 1
            self.logger.debug(start_idx, "fixed to 1 for robot", robot_id)

            reachable = self.reachable_for_window(
                robot, start_node, start, end, bfs_variant
            )

            if goal_node in reachable.get(start + 1, set()):
                self.logger.standard(
                    f"Goal is reachable at timestep 1 for robot {robot_id}. Fixing instantaneous path."
                )
                active_cells[(robot_id, start)] = [start_node]
                for t in range(start + 1, end):
                    goal_idx = goal_node + (self.num_nodes * t) + robot_offset
                    fixed_ones[goal_idx] = 1
                    active_cells[(robot_id, t)] = [goal_node]
            else:
                active_cells[(robot_id, start)] = [start_node]
                for t in range(start + 1, end):
                    active_cells[(robot_id, t)] = list(reachable.get(t, {goal_node}))

        return fixed_ones, active_cells

    def apply_one_hot(self):
        """Apply one-hot constraint: exactly one node per time step per robot."""
        K_hot = self.penalties["K_hot"]
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)

            start, end = self.window_steps(robot_id)

            for t in range(start, end):
                indices = [
                    node_id + (self.num_nodes * t) + robot_offset
                    for node_id in self._nodes(robot_id, t)
                ]

                for n in indices:
                    self.Q[(n, n)] = self.Q.get((n, n), 0) - K_hot

                for i, n in enumerate(indices):
                    for m in indices[i + 1 :]:
                        self.Q[(n, m)] = self.Q.get((n, m), 0) + 2 * K_hot

    def apply_start_penalty(self):
        """Apply start node penalty: must start at the given node for each robot."""
        K_start = self.penalties["K_start"]
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)

            start_time = self.problem.robots[robot_id].start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            # Get start node for this robot (assuming it's stored as node index)
            start_node = self.problem.get_graph_robot_current_goal(robot_id)[0]
            start_idx = start_node + (start * self.num_nodes) + robot_offset
            self.Q[(start_idx, start_idx)] = (
                self.Q.get((start_idx, start_idx), 0) - K_start
            )

    def apply_lock_after_goal(self):
        """Apply lock-after-goal constraint: once at goal, stay there for each robot."""
        K_lock = self.penalties["K_lock"]
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)

            start, end = self.window_steps(robot_id)

            # Get goal node for this robot
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]

            for t in range(start, end - 1):
                if goal_node not in self._nodes(robot_id, t):
                    continue
                goal_idx_t = goal_node + (self.num_nodes * t) + robot_offset
                # K_lock * x_goal,t * (sum of the other nodes at t+1): under
                # one-hot the same as K_lock * x_goal,t * (1 - x_goal,t+1), but
                # with no diagonal, which the diagonal fixer would read as "the
                # goal is the most expensive node" (see GridQUBOBuilder)
                for node in self._nodes(robot_id, t + 1):
                    if node == goal_node:
                        continue
                    m = node + (self.num_nodes * (t + 1)) + robot_offset
                    self.Q[(goal_idx_t, m)] = self.Q.get((goal_idx_t, m), 0) + K_lock

    def apply_adjacency_reward(self):
        """Apply adjacency reward: encourage moving to adjacent nodes for each robot."""
        K_adj = self.penalties["K_adj"]
        adjacency = self.graph.adjacency
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)

            start, end = self.window_steps(robot_id)
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]

            for t in range(start, end - 1):
                next_active = set(self._nodes(robot_id, t + 1))
                for node_i in self._nodes(robot_id, t):
                    n = node_i + (self.num_nodes * t) + robot_offset
                    self.Q[(n, n)] = self.Q.get((n, n), 0) + K_adj
                    targets = adjacency[node_i]
                    # Staying on the goal is rewarded like a move; elsewhere a
                    # stay keeps paying K_adj (see GridQUBOBuilder)
                    if node_i == goal_node:
                        targets = [*targets, (node_i, 1.0)]
                    for node_j, weight in targets:
                        if self._active_cells is not None and node_j not in next_active:
                            continue
                        m = node_j + (self.num_nodes * (t + 1)) + robot_offset
                        self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj * weight

    def goal_distance_map(self, robot_id):
        """{node: steps} shortest path in edges from every node to the robot's
        goal (BFS backwards over graph.adjacency; one edge per timestep, so
        weights don't enter). Nodes that can't reach the goal are absent."""
        if robot_id not in self._goal_distance_maps:
            goal = self.problem.get_graph_robot_current_goal(robot_id)[1]
            dist = {goal: 0}
            frontier = deque([goal])
            while frontier:
                node = frontier.popleft()
                for nxt in self._neighbour_nodes(node):
                    if nxt not in dist:
                        dist[nxt] = dist[node] + 1
                        frontier.append(nxt)
            self._goal_distance_maps[robot_id] = dist
        return self._goal_distance_maps[robot_id]

    def apply_goal_objective(self):
        """
        Arrival-time goal term, the graph counterpart of
        GridQUBOBuilder.apply_goal_objective(): -w on the goal node at every
        step, plus the distance left (goal_distance_map) priced at w -- spread
        over the window's steps ("shaped") or on the last step ("terminal"),
        and K_goal * distance on the robot's last step if its deadline falls
        in the window.
        """
        robot_nums = self.problem.get_robot_nums()
        K_goal = self.penalties.get("K_goal", 0)
        w = self.goal_weight

        for robot_id in self.get_active_robot_in_window():
            robot = self.problem.robots[robot_id]
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            start, end = self.window_steps(robot_id)
            deadline_in_window = (
                robot.start_time + robot.T <= self.current_T + self.t_max
            )
            steps = list(range(start + 1, end))
            if not steps:
                continue
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]
            dist = self.goal_distance_map(robot_id)
            unreachable = max(dist.values(), default=0) + 1

            if self.goal_objective == "shaped":
                ramp = {t: t - start for t in steps}
                total = sum(ramp.values())
                coeff = {t: w * ramp[t] / total for t in steps}
            else:
                coeff = {steps[-1]: w}
            if deadline_in_window:
                coeff[steps[-1]] = coeff.get(steps[-1], 0) + K_goal

            for t in steps:
                for node in self._nodes(robot_id, t):
                    n = node + (self.num_nodes * t) + robot_offset
                    e = coeff.get(t, 0) * dist.get(node, unreachable)
                    if node == goal_node:
                        e -= w
                    if e:
                        self.Q[(n, n)] = self.Q.get((n, n), 0) + e

    def apply_crash_penalty(self):
        K_crash = self.penalties.get("K_crash", 0)
        robot_nums = self.problem.get_robot_nums()
        active_robots_per_timestep = self.get_active_robots_per_timestep_in_window()
        for t, active_robots in active_robots_per_timestep.items():
            if len(active_robots) < 2:
                continue
            t_window = t - self.current_T
            for robot_id1 in active_robots:
                for robot_id2 in active_robots:
                    if robot_nums[robot_id1] >= robot_nums[robot_id2]:
                        continue
                    robot_offset1 = robot_nums[robot_id1] * (
                        self.num_nodes * self.total_t
                    )
                    robot_offset2 = robot_nums[robot_id2] * (
                        self.num_nodes * self.total_t
                    )
                    shared_nodes = set(self._nodes(robot_id1, t_window)) & set(
                        self._nodes(robot_id2, t_window)
                    )
                    for node_i in shared_nodes:
                        idx1 = node_i + (self.num_nodes * t_window) + robot_offset1
                        idx2 = node_i + (self.num_nodes * t_window) + robot_offset2
                        self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_crash

    def apply_trailing_penalty(self):
        """
        Robot-robot separation penalty.
        Graph mode stays exact-cell/exact-swap only (no D_ab shell): a graph's node `pos`
        isn't guaranteed to be a uniform grid, same reasoning CBS/ILP already
        apply to graph mode.

        P = K_crash * r1[t]*r2[t] + K_trail * (r1[t+1]*r2[t] + r1[t]*r2[t+1])

        Summed over nodes shared by two robots, this penalizes both same-
        node/same-time occupancy (the crash case) and the cross-time
        trailing hazard. The two are weighted independently: the same-node/
        same-time term reuses K_crash as-is (identical to apply_crash_penalty,
        so crash protection doesn't get overshadowed), while the cross-time terms
        get their own K_trail.
        Those cross-time terms also fire when one robot simply follows another
        into a just-vacated node, which isn't an actual collision (an
        accepted overconstraint), so K_trail is the knob to tune down if that
        false-positive cost outweighs the benefit, independently of crash
        protection.

        When K_trail is active it replaces apply_crash_penalty
        (see build()) so the same-time term isn't double counted.
        """
        K_crash = self.penalties.get("K_crash", 0)
        K_trail = self.penalties.get("K_trail", 0)
        robot_nums = self.problem.get_robot_nums()
        active_robots_per_timestep = self.get_active_robots_per_timestep_in_window()

        for t, active_robots in active_robots_per_timestep.items():
            if len(active_robots) < 2:
                continue
            t_window = t - self.current_T
            next_active_robots = active_robots_per_timestep.get(t + 1, [])

            for robot_id1 in active_robots:
                for robot_id2 in active_robots:
                    if robot_nums[robot_id1] >= robot_nums[robot_id2]:
                        continue

                    robot_offset1 = robot_nums[robot_id1] * (
                        self.num_nodes * self.total_t
                    )
                    robot_offset2 = robot_nums[robot_id2] * (
                        self.num_nodes * self.total_t
                    )

                    # Same-node, same-time term (native crash constraint)
                    shared_nodes_t = set(self._nodes(robot_id1, t_window)) & set(
                        self._nodes(robot_id2, t_window)
                    )
                    for node_i in shared_nodes_t:
                        idx1 = node_i + (self.num_nodes * t_window) + robot_offset1
                        idx2 = node_i + (self.num_nodes * t_window) + robot_offset2
                        self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_crash

                    # Cross-time swap terms, only meaningful if both robots are
                    # still active in the window at t+1
                    if (
                        robot_id1 not in next_active_robots
                        or robot_id2 not in next_active_robots
                    ):
                        continue
                    t_next_window = t_window + 1

                    # robot1 arrives at t+1 where robot2 was at t
                    shared_r1_next = set(self._nodes(robot_id1, t_next_window)) & set(
                        self._nodes(robot_id2, t_window)
                    )
                    for node_i in shared_r1_next:
                        idx1 = node_i + (self.num_nodes * t_next_window) + robot_offset1
                        idx2 = node_i + (self.num_nodes * t_window) + robot_offset2
                        self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_trail

                    # robot2 arrives at t+1 where robot1 was at t
                    shared_r2_next = set(self._nodes(robot_id1, t_window)) & set(
                        self._nodes(robot_id2, t_next_window)
                    )
                    for node_i in shared_r2_next:
                        idx1 = node_i + (self.num_nodes * t_window) + robot_offset1
                        idx2 = node_i + (self.num_nodes * t_next_window) + robot_offset2
                        self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_trail

    def reachable_positions_safe(self, robot, start_node, start_time, end_time):
        """Monotone reachability over nodes; see preprocessing.reachable_safe."""
        return reachable_safe(start_node, self._neighbour_nodes, start_time, end_time)

    def _neighbour_nodes(self, node):
        """graph.adjacency holds (neighbour, weight) pairs; reachability
        only needs the neighbours."""
        return (n for n, _ in self.graph.adjacency.get(node, ()))

    def reachable_positions_aggressive(
        self, robot, start_node, start_time, end_time, blocked=None, allow_wait_at=None
    ):
        """Nodes reachable per step without revisiting; see
        preprocessing.reachable_aggressive for blocked / allow_wait_at."""
        goal_node = self.problem.get_graph_robot_current_goal(robot.robot_id)[1]
        return reachable_aggressive(
            start_node,
            goal_node,
            self._neighbour_nodes,
            start_time,
            end_time,
            blocked,
            allow_wait_at,
        )
