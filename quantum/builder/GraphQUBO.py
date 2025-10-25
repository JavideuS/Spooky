from .base_qubo import BaseQUBO


class GraphQUBO(BaseQUBO):
    def __init__(
        self,
        problem,
        penalties,
        name="graph",
        var_limit=131,
        window_max_steps=None,
        distance_scaling=None,
    ):
        self.graph = problem.graph
        self.num_nodes = len(self.graph.nodes)
        super().__init__(
            problem,
            penalties,
            name=name,
            var_limit=var_limit,
            window_max_steps=window_max_steps,
            distance_scaling=distance_scaling,
        )
        # Multi-robot support: calculate total variables for all robots
        self.initial_num_vars = (self.num_nodes * self.T *
                                 self.problem.num_robots)

    def build(self, constraints_to_apply=None):
        """Build the QUBO dictionary for graph-based pathfinding."""
        if constraints_to_apply is None:
            penalty_to_constraint = {
                "K_hot": "one_hot",
                "K_adj": "adjacency_reward",
                "K_start": "start",
                "K_goal": "goal",
                "K_lock": "lock",
                "K_bt": "backtracking",
            }
            constraints_to_apply = [
                v for k, v in penalty_to_constraint.items()
                if k in self.penalties
            ]
        
        self.Q = {}

        if "one_hot" in constraints_to_apply:
            self.apply_one_hot()
        if "start" in constraints_to_apply:
            self.apply_start_penalty()
        if "goal" in constraints_to_apply:
            self.apply_goal_penalty()
        if "adjacency" in constraints_to_apply:
            self.apply_adjacency_constraint()
        if "lock" in constraints_to_apply:
            self.apply_lock_after_goal()
        if "adjacency_reward" in constraints_to_apply:
            self.apply_adjacency_reward()
        if "backtracking" in constraints_to_apply:
            self.apply_backtracking_penalty()
        if "multi_robot_collision" in constraints_to_apply:
            self.apply_multi_robot_collision_penalty()
        if "multi_robot_proximity" in constraints_to_apply:
            self.apply_multi_robot_proximity_penalty()

        return self.Q

    def apply_one_hot(self):
        """Apply one-hot constraint: exactly one node per time step per robot."""
        K_hot = self.penalties['K_hot']

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (self.num_nodes * self.T)
            robot = self.problem.robots[robot_id]
            start_time = robot.start_time
            end_time = robot.T + start_time

            for t in range(start_time, end_time):
                # All node variables at time t for this robot
                # Formula: node_id + num_nodes * t + robot_offset
                indices = [node_id + (self.num_nodes * t) + robot_offset
                           for node_id in range(self.num_nodes)]

                for n in indices:
                    self.Q[(n, n)] = self.Q.get((n, n), 0) - K_hot

                for i, n in enumerate(indices):
                    for m in indices[i + 1:]:
                        self.Q[(n, m)] = self.Q.get((n, m), 0) + 2 * K_hot

    def apply_start_penalty(self):
        """Apply start node penalty: must start at the given node for each robot."""
        K_start = self.penalties['K_start']

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (self.num_nodes * self.T)
            start_time = self.problem.robots[robot_id].start_time

            # Get start node for this robot (assuming it's stored as node index)
            start_node = self.problem.get_graph_robot_current_goal(robot_id)[0]
            start_idx = start_node + (start_time * self.num_nodes) + robot_offset
            self.Q[(start_idx, start_idx)] = (
                self.Q.get((start_idx, start_idx), 0) - K_start
            )

    def apply_goal_penalty(self):
        """Apply goal node penalty: encourage reaching the goal for each robot."""
        K_goal = self.penalties['K_goal']

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            robot_offset = robot_num * (self.num_nodes * self.T)
            start_time = robot.start_time
            end_time = robot.T + start_time

            # Get goal node for this robot
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]

            for t in range(start_time + 1, end_time):
                goal_idx = goal_node + (self.num_nodes * t) + robot_offset
                time_factor = 1 + ((t - start_time) / (end_time - start_time))
                self.Q[(goal_idx, goal_idx)] = (
                    self.Q.get((goal_idx, goal_idx), 0) - K_goal * time_factor
                )

    def apply_lock_after_goal(self):
        """Apply lock-after-goal constraint: once at goal, stay there for each robot."""
        K_lock = self.penalties['K_lock']

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            robot_offset = robot_num * (self.num_nodes * self.T)
            start_time = robot.start_time
            end_time = robot.T + start_time

            # Get goal node for this robot
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]

            for t in range(start_time, end_time - 1):
                goal_idx_t = goal_node + (self.num_nodes * t) + robot_offset
                goal_idx_t1 = goal_node + (self.num_nodes * (t + 1)) + robot_offset
                self.Q[(goal_idx_t, goal_idx_t)] = (
                    self.Q.get((goal_idx_t, goal_idx_t), 0) + K_lock
                )
                self.Q[(goal_idx_t, goal_idx_t1)] = (
                    self.Q.get((goal_idx_t, goal_idx_t1), 0) - K_lock
                )

    def apply_adjacency_constraint(self):
        """
        Apply adjacency constraint: only move between connected nodes for each robot.
        It enforces edge movements and penalizes non-adjacent moves.
        """
        K_adj = self.penalties['K_adj']
        adjacency = self.graph.adjacency

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            robot_offset = robot_num * (self.num_nodes * self.T)
            start_time = robot.start_time
            end_time = robot.T + start_time

            for t in range(start_time, end_time - 1):
                for node_i in range(self.num_nodes):
                    n = node_i + (self.num_nodes * t) + robot_offset
                    # Skip if no connections
                    # if node_i not in adjacency:
                    #     continue

                    # Reward staying at connected nodes
                    for (node_j, weight) in adjacency[node_i]:
                        m = node_j + (self.num_nodes * (t + 1)) + robot_offset
                        self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj * weight

                    # Penalize moving to non-adjacent nodes
                    for node_j in range(self.num_nodes):
                        if (node_j != node_i and
                            (node_j, 1.0) not in adjacency[node_i]):
                            m = node_j + (self.num_nodes * (t + 1)) + robot_offset
                            self.Q[(n, m)] = self.Q.get((n, m), 0) + K_adj

    def apply_adjacency_reward(self):
        """Apply adjacency reward: encourage moving to adjacent nodes for each robot."""
        K_adj = self.penalties['K_adj']
        adjacency = self.graph.adjacency

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            robot_offset = robot_num * (self.num_nodes * self.T)
            start_time = robot.start_time
            end_time = robot.T + start_time

            for t in range(start_time, end_time - 1):
                for node_i in range(self.num_nodes):
                    n = node_i + (self.num_nodes * t) + robot_offset
                    # Skip if no connections
                    # if node_i not in adjacency:
                    #     continue

                    self.Q[(n, n)] = self.Q.get((n, n), 0) + K_adj

                    # Reward staying at connected nodes
                    for (node_j, weight) in adjacency[node_i]:
                        m = node_j + (self.num_nodes * (t + 1)) + robot_offset
                        self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj * weight

    def apply_backtracking_penalty(self):
        """Apply backtracking penalty: discourage revisiting nodes for each robot."""
        K_bt = self.penalties['K_bt']

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            robot_offset = robot_num * (self.num_nodes * self.T)
            start_time = robot.start_time
            end_time = robot.T + start_time

            # Get goal node for this robot
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]

            for node_i in range(self.num_nodes):
                # Skip goal node - allow multiple visits
                if node_i == goal_node:
                    continue

                # For all time pairs t1 < t2
                for t1 in range(start_time, end_time):
                    n1 = node_i + (self.num_nodes * t1) + robot_offset
                    for t2 in range(t1 + 1, end_time):
                        n2 = node_i + (self.num_nodes * t2) + robot_offset
                        self.Q[(n1, n2)] = self.Q.get((n1, n2), 0) + K_bt

            # Note that this will probably make them be removed by the pre-processing
            # To try avoid that added weights to high penalize early positions and softly penalize early ones
            # But not sure yet if the soft penalty is small enough to be not removed in cases where backtracking is needed
            if robot.active and robot.path:
                len_sol = len(robot.path)
                for t in range(start_time, end_time):
                    for p_idx, pos in enumerate(robot.path):
                        node_id = self.graph.get_node_from_position(pos[0][:2])
                        n = node_id + (self.num_nodes * t) + robot_offset
                        time_factor = (1 + (len_sol - p_idx)) / len_sol
                        self.Q[(n, n)] = self.Q.get((n, n), 0) + K_bt * time_factor

    def get_fixed_variables(self):
        """Identify variables that can be fixed based on current problem state for all robots."""
        fixed = {}

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            robot_offset = robot_num * (self.num_nodes * self.T)
            start_time = robot.start_time
            end_time = robot.T + start_time

            # Fix start node at start_time for this robot
            start_node = self.problem.get_graph_robot_current_goal(robot_id)[0]
            start_idx = start_node + (start_time * self.num_nodes) + robot_offset
            fixed[start_idx] = 1

            # Fix all other nodes at start_time to 0 for this robot
            for node_i in range(self.num_nodes):
                if node_i != start_node:
                    n = node_i + (self.num_nodes * start_time) + robot_offset
                    fixed[n] = 0

            # Initialize reachable set with the start node at start_time
            reachable_at_time = {start_time: {start_node}}

            # Perform a BFS-like traversal to find all reachable nodes at each time step
            for t in range(start_time, end_time - 1):
                reachable_at_time[t + 1] = set()
                for node_i in reachable_at_time[t]:
                    for node_j, _ in self.graph.adjacency.get(node_i, []):
                        reachable_at_time[t + 1].add(node_j)

            # Fix unreachable nodes to 0 for all time steps for this robot
            for t in range(start_time + 1, end_time):
                for node_id in range(self.num_nodes):
                    var_idx = node_id + (self.num_nodes * t) + robot_offset
                    if node_id not in reachable_at_time.get(t, set()):
                        fixed[var_idx] = 0

        return fixed

    # def apply_multi_robot_collision_penalty(self):
    #     """Apply collision avoidance penalty: prevent robots from occupying the same node at the same time."""
    #     K_crash = self.penalties.get('K_crash', 5)

    #     # For each time step, check all pairs of robots
    #     for t in range(self.T):
    #         for node_id in range(self.num_nodes):
    #             # Get all robot pairs that could collide at this node and time
    #             for r1_num in range(self.problem.num_robots):
    #                 for r2_num in range(r1_num + 1, self.problem.num_robots):
    #                     robot1_offset = r1_num * (self.num_nodes * self.T)
    #                     robot2_offset = r2_num * (self.num_nodes * self.T)

    #                     idx1 = node_id + (self.num_nodes * t) + robot1_offset
    #                     idx2 = node_id + (self.num_nodes * t) + robot2_offset

    #                     # Add penalty for both robots being at the same node
    #                     self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_crash

    # def apply_multi_robot_proximity_penalty(self):
    #     """Apply proximity penalty: discourage robots from being too close to each other."""
    #     K_proximity = self.penalties.get('K_proximity', 2)

    #     # For each time step, check robot proximity
    #     for t in range(self.T):
    #         for r1_num in range(self.problem.num_robots):
    #             for r2_num in range(r1_num + 1, self.problem.num_robots):
    #                 robot1_offset = r1_num * (self.num_nodes * self.T)
    #                 robot2_offset = r2_num * (self.num_nodes * self.T)

    #                 # Check all node pairs for proximity
    #                 for node1_id in range(self.num_nodes):
    #                     for node2_id in range(self.num_nodes):
    #                         # Skip if same node (handled by collision penalty)
    #                         if node1_id == node2_id:
    #                             continue

    #                         # Check if nodes are adjacent (proximity)
    #                         is_adjacent = False
    #                         for (adj_node, _) in self.graph.adjacency.get(node1_id, []):
    #                             if adj_node == node2_id:
    #                                 is_adjacent = True
    #                                 break

    #                         if is_adjacent:
    #                             idx1 = node1_id + (self.num_nodes * t) + robot1_offset
    #                             idx2 = node2_id + (self.num_nodes * t) + robot2_offset

    #                             # Add penalty for robots being adjacent
    #                             self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_proximity

    def max_window_size(self):
        """Calculate max window size for graph-based problems."""
        return max(1, self.var_limit // (self.num_nodes * self.problem.num_robots))
