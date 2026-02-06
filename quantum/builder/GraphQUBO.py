from .base_qubo import BaseQUBO
import numpy as np


class GraphQUBO(BaseQUBO):
    def __init__(
        self,
        problem,
        penalties,
        name="graph",
        var_limit=1201,  #65 131
        window_max_steps=None,
        distance_scaling="enhanced_linear",
        robot_window_limits=None,
        verbose_level=2,
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
            robot_window_limits=robot_window_limits,
            verbose_level=verbose_level,
        )
        # Multi-robot support: calculate total variables for all robots
        # Use problem.T (total timeline) not total_t (window size) to match QUBOBuilder
        self.initial_num_vars = (self.num_nodes * self.problem.T *
                                 self.problem.num_robots)
        
        # Compute goal-oriented connectivity potential for each robot
        # This helps avoid dead-ends by identifying nodes where neighbors lead away from goal
        # Store per-robot potentials for multi-robot scenarios
        self.P_connectivity_per_robot = {}
        self.P_obs = self.compute_spatial_obstacle_potential()
        
        self.logger.standard("Window max steps:", self.max_window_size())

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
                "K_crash": "multi_robot_collision",
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
            self.apply_multi_robot_penalty()
        if "multi_robot_proximity" in constraints_to_apply:
            self.apply_multi_robot_proximity_penalty()

        return self.Q

    def apply_one_hot(self):
        """Apply one-hot constraint: exactly one node per time step per robot."""
        K_hot = self.penalties['K_hot']
        robot_nums = self.problem.get_robot_nums()
        
        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            # Debug logs for empty QUBO diagnosis
            # self.logger.debug(f"Robot {robot_id}: start={start}, end={end}, start_time={start_time}, T={robot.T}, current_T={self.current_T}, t_max={self.t_max}")

            for t in range(start, end):
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

    def apply_goal_penalty(self):
        """Apply goal node penalty: encourage reaching the goal for each robot."""
        K_goal = self.penalties['K_goal']
        robot_nums = self.problem.get_robot_nums()
        
        for robot_id in self.get_active_robot_in_window():
            robot = self.problem.robots[robot_id]
            start_time = robot.start_time
            end_time = robot.T + start_time

            if end_time > self.current_T + self.t_max:
                # Goal not reachable in this window, use approximation
                self.apply_goal_approximation_penalty(robot_id)
            else:
                # Goal is reachable, apply standard goal penalty
                robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
                
                start = 0
                if self.current_T < start_time:
                    start = start_time - self.current_T

                end = end_time - self.current_T
                goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]
                window_constant = 1 + (0.6 / end)

                for t in range(start + 1, end):
                    goal_idx = goal_node + (self.num_nodes * t) + robot_offset
                    time_factor = 1 + ((t - start) / (end - start))
                    self.Q[(goal_idx, goal_idx)] = (
                        self.Q.get((goal_idx, goal_idx), 0) - K_goal * time_factor * window_constant
                    )


    def apply_lock_after_goal(self):
        """Apply lock-after-goal constraint: once at goal, stay there for each robot."""
        K_lock = self.penalties['K_lock']
        robot_nums = self.problem.get_robot_nums()
        
        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            # Get goal node for this robot
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]

            for t in range(start, end - 1):
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
        robot_nums = self.problem.get_robot_nums()
        
        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            for t in range(start, end - 1):
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
        robot_nums = self.problem.get_robot_nums()
        
        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            for t in range(start, end - 1):
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

    def calculate_euclidean_penalty(self, raw_dist, K_goal_approx, time_factor):
        """
        Calculate Euclidean distance penalty using the specified scaling method.
        
        Args:
            raw_dist: Raw Euclidean distance
            K_goal_approx: Goal approximation penalty coefficient
            time_factor: Time-based scaling factor
            
        Returns:
            K_dis: Calculated distance penalty
        """
        if self.distance_scaling == "enhanced_linear":
            dist_to_goal = raw_dist * 0.165
            K_dis = K_goal_approx * (1/(0.7 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "exponential":
            dist_to_goal = raw_dist * 1.2
            K_dis = K_goal_approx * (1 / (1 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "quadratic":
            dist_to_goal = raw_dist ** 1.3
            K_dis = K_goal_approx * (1/(1 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "logarithmic":
            import numpy as np
            dist_to_goal = np.log(1 + raw_dist * 2)
            K_dis = K_goal_approx * (1/(1 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "adaptive":
            num_nodes = self.num_nodes
            if num_nodes <= 9:
                dist_to_goal = raw_dist * 0.4
                K_dis = K_goal_approx * (1/(0.2 + dist_to_goal)) * time_factor
            elif num_nodes <= 25:
                dist_to_goal = raw_dist * 0.8
                K_dis = K_goal_approx * (1/(0.4 + dist_to_goal)) * time_factor
            else:
                dist_to_goal = raw_dist * 1.2
                K_dis = K_goal_approx * (1/(0.8 + dist_to_goal)) * time_factor
                
        else:
            dist_to_goal = raw_dist * 2
            K_dis = K_goal_approx * (1 / (1 + dist_to_goal)) * time_factor
            
        return K_dis

    def _compute_node_connectivity_potential(self, goal_node=None, start_node=None):
        """
        Compute goal-oriented and directionally-aware connectivity potential for each node.
        
        This measures TWO factors:
        1. How many neighbors move you CLOSER to the goal (goal-oriented)
        2. How many neighbors are DIRECTIONALLY ALIGNED with the path from start to goal
        
        Nodes whose neighbors point away from the start→goal direction get higher potential.
        
        Args:
            goal_node: Target node ID (if None, uses simple degree-based)
            start_node: Current position node ID (if None, ignores directional factor)
        
        Returns:
            Array of potential values, one per node.
        """
        P_nodes = np.zeros(self.num_nodes)
        
        # If no specific goal, compute general connectivity
        if goal_node is None:
            # Use simple degree-based for general case
            degrees = np.zeros(self.num_nodes)
            for node_id in range(self.num_nodes):
                degrees[node_id] = len(self.graph.adjacency.get(node_id, []))
            
            max_degree = max(degrees) if max(degrees) > 0 else 1
            
            for node_id in range(self.num_nodes):
                normalized_degree = degrees[node_id] / max_degree
                P_nodes[node_id] = np.exp(-(normalized_degree ** 2))
        else:
            # Goal-oriented: count neighbors that move you CLOSER to goal
            goal_pos = self.graph.get_node_position(goal_node)
            if goal_pos is None:
                return P_nodes  # Fallback to zeros
            
            # Get start position if provided (for directional alignment)
            start_pos = None
            if start_node is not None:
                start_pos = self.graph.get_node_position(start_node)
            
            for node_id in range(self.num_nodes):
                node_pos = self.graph.get_node_position(node_id)
                if node_pos is None:
                    continue
                
                # Factor 1: Goal-oriented connectivity
                # Distance from current node to goal
                current_dist = self.problem.euclidean_distance(node_pos, goal_pos)
                
                # Count neighbors and how many lead closer to goal
                neighbors = self.graph.adjacency.get(node_id, [])
                total_neighbors = len(neighbors)
                
                if total_neighbors == 0:
                    P_nodes[node_id] = 1.0  # Isolated node = maximum penalty
                    continue
                
                # Count "good" neighbors (those closer to goal than current node)
                good_neighbors = 0
                for neighbor_id, _ in neighbors:
                    neighbor_pos = self.graph.get_node_position(neighbor_id)
                    if neighbor_pos is not None:
                        neighbor_dist = self.problem.euclidean_distance(neighbor_pos, goal_pos)
                        if neighbor_dist < current_dist:
                            good_neighbors += 1
                
                # Heuristic: Inverse of (1 + good_neighbors)
                # 0 good neighbors (dead end) -> 1/1 = 1.0
                # 1 good neighbor (narrow)    -> 1/2 = 0.5
                # 2 good neighbors            -> 1/3 = 0.33
                # 3 good neighbors            -> 1/4 = 0.25
                P_nodes[node_id] = 1.0 / (1.0 + good_neighbors)

        
        return P_nodes
    
    def compute_spatial_obstacle_potential(self, sigma=1.5, isolation_threshold=3):
        """
        Compute obstacle potential using node positions and connectivity patterns.
        Nodes with few connections are likely near obstacles/boundaries and create
        repulsive potential fields similar to the grid-based approach.
        
        Args:
            sigma: Controls spatial decay (higher = wider influence)
            isolation_threshold: Nodes with fewer neighbors are considered near obstacles
        
        Returns:
            Array of potential values per node
        """
        P_nodes = np.zeros(self.num_nodes)
        
        # Collect positions and identify "obstacle-adjacent" nodes
        obstacle_nodes = []
        all_positions = {}
        
        for node_id in range(self.num_nodes):
            pos = self.graph.get_node_position(node_id)
            if pos is not None:
                all_positions[node_id] = pos
                
                # Low connectivity suggests proximity to obstacles
                neighbors = self.graph.adjacency.get(node_id, [])
                if len(neighbors) <= isolation_threshold:
                    obstacle_nodes.append((node_id, pos))
        
        if len(obstacle_nodes) == 0 or len(all_positions) == 0:
            return P_nodes
        
        # Compute potential: each obstacle node creates Gaussian repulsion
        for node_id, node_pos in all_positions.items():
            potential = 0.0
            
            for obs_id, obs_pos in obstacle_nodes:
                if node_id == obs_id:
                    # Self-contribution (node is itself near obstacle)
                    potential += 1.0
                else:
                    # Spatial distance-based Gaussian decay
                    dist = self.problem.euclidean_distance(node_pos, obs_pos)
                    potential += np.exp(-dist**2 / (2 * sigma**2))
            
            P_nodes[node_id] = potential
        
        # Normalize to [0, 1]
        if np.max(P_nodes) > 0:
            P_nodes /= np.max(P_nodes)
        
        return P_nodes
    
    def apply_goal_approximation_penalty(self, robot_id):
        """
        Apply goal approximation penalty using Euclidean distance heuristic.
        Encourages getting closer to the goal when it cannot be reached in current window.
        Includes goal-oriented connectivity potential to avoid dead-ends.
        """
        K_goal_approx = self.penalties.get('K_goal_approx', 0.5)
        if K_goal_approx == 0:
            return
        
        K_deadend_repel = 0.4  # Dead-end repulsion strength (for low-connectivity nodes)
        K_repel = 1.0 # Probably requires higher tuning, still need to ponder if higher means better
        

        robot_nums = self.problem.get_robot_nums()
        robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
        robot = self.problem.robots[robot_id]
        
        start_time = robot.start_time
        start = 0
        if self.current_T < start_time:
            start = start_time - self.current_T
        
        goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]
        goal_pos = self.graph.get_node_position(goal_node)
        
        # Get current position (start node for this window)
        start_node = self.problem.get_graph_robot_current_goal(robot_id)[0]
        
        # Compute goal-oriented and proximity-aware connectivity potential for this robot
        # Cache it to avoid recomputation
        if robot_id not in self.P_connectivity_per_robot:
            self.P_connectivity_per_robot[robot_id] = self._compute_node_connectivity_potential(
                goal_node, start_node
            )
        
        for t in range(start + 1, self.t_max):
            time_factor = (1.2) ** (5 * (t - start) / (self.t_max - start))
            
            for node_id in range(self.num_nodes):
                node_pos = self.graph.get_node_position(node_id)
                if node_pos is None or goal_pos is None:
                    continue
                
                # Goal attraction (distance-based)
                raw_dist = self.problem.euclidean_distance(node_pos, goal_pos)
                K_dis = self.calculate_euclidean_penalty(raw_dist, K_goal_approx, time_factor)
                
                # Dead-end repulsion (goal-oriented connectivity)
                # Penalize nodes with lower connectivity towards goal
                # P value is 1.0 (dead end), 0.5 (1 neighbor), 0.33 (2 neighbors)...
                # We apply the penalty scaled by this value
                K_repel_val = self.penalties.get('K_deadend_repel', 0.2)
                K_deadend = K_repel_val * self.P_connectivity_per_robot[robot_id][node_id]
                
                K_obs = K_repel #* self.P_obs[node_id] # This is environment-oriented (avoid obstacles)
                
                var_idx = node_id + (self.num_nodes * t) + robot_offset
                # Apply both goal attraction and dead-end repulsion
                self.Q[(var_idx, var_idx)] = self.Q.get((var_idx, var_idx), 0) - K_dis + K_obs + K_deadend

    def apply_backtracking_penalty(self):
        """Apply backtracking penalty: discourage revisiting nodes for each robot."""
        K_bt = self.penalties['K_bt']
        robot_nums = self.problem.get_robot_nums()
        
        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            # Get goal node for this robot
            goal_node = self.problem.get_graph_robot_current_goal(robot_id)[1]

            for node_i in range(self.num_nodes):
                # Skip goal node - allow multiple visits
                if node_i == goal_node:
                    continue

                # For all time pairs t1 < t2
                for t1 in range(start, end):
                    n1 = node_i + (self.num_nodes * t1) + robot_offset
                    for t2 in range(t1 + 1, end):
                        n2 = node_i + (self.num_nodes * t2) + robot_offset
                        self.Q[(n1, n2)] = self.Q.get((n1, n2), 0) + K_bt

            # Note that this will probably make them be removed by the pre-processing
            # To try avoid that added weights to high penalize early positions and softly penalize early ones
            # But not sure yet if the soft penalty is small enough to be not removed in cases where backtracking is needed
            if robot.active and robot.path:
                len_sol = len(robot.path)
                for t in range(start, end):
                    for p_idx, pos in enumerate(robot.path):
                        node_id = self.graph.get_node_from_position(pos[:2])
                        n = node_id + (self.num_nodes * t) + robot_offset
                        time_factor = (1 + (len_sol - p_idx)) / len_sol
                        self.Q[(n, n)] = self.Q.get((n, n), 0) + K_bt * time_factor
    
    def apply_multi_robot_penalty(self):
        K_crash = self.penalties.get('K_crash', 0)
        robot_nums = self.problem.get_robot_nums()
        active_robots_per_timestep = self.get_active_robots_per_timestep_in_window()
        for t, active_robots in active_robots_per_timestep.items():
            if len(active_robots) < 2:
                continue  # no collision possible
            for node_i in range(self.num_nodes):
                for robot_id1 in active_robots:
                    for robot_id2 in active_robots:
                        if robot_nums[robot_id1] >= robot_nums[robot_id2]:
                            continue
                        robot_offset1 = robot_nums[robot_id1] * (self.num_nodes * self.total_t)
                        robot_offset2 = robot_nums[robot_id2] * (self.num_nodes * self.total_t)
                        idx1 = node_i + (self.num_nodes * (t - self.current_T)) + robot_offset1
                        idx2 = node_i + (self.num_nodes * (t - self.current_T)) + robot_offset2
                        self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_crash

    def get_fixed_variables(self):
        """Identify variables that can be fixed based on current problem state for all robots."""
        fixed = {}
        robot_nums = self.problem.get_robot_nums()
        
        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (self.num_nodes * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            # Fix start node at start_time for this robot
            start_node, goal_node = self.problem.get_graph_robot_current_goal(robot_id)
            start_idx = start_node + (start * self.num_nodes) + robot_offset
            fixed[start_idx] = 1

            # Fix all other nodes at start_time to 0 for this robot
            for node_i in range(self.num_nodes):
                if node_i != start_node:
                    n = node_i + (self.num_nodes * start) + robot_offset
                    fixed[n] = 0

            # Aggresive version
            reachable_at_time = self.reachable_positions_aggressive(robot, start_node, start, end)
            
            # Check if goal is reachable at timestep 1 (one movement away)
            # If so, fix the entire instantaneous path
            if goal_node in reachable_at_time.get(start + 1, set()):
                self.logger.debug(f"Goal is reachable at timestep 1 for robot {robot_id}. Fixing instantaneous path.")
                
                # Fix timestep start + 1: goal = 1, all others = 0
                for node_id in range(self.num_nodes):
                    n = node_id + (self.num_nodes * (start + 1)) + robot_offset
                    if node_id == goal_node:
                        fixed[n] = 1
                        self.logger.debug(f"  Fixed goal node {n} at node {node_id} to 1 at timestep {start + 1}")
                    else:
                        fixed[n] = 0
                
                # Fix all subsequent timesteps: robot stays at goal
                for t in range(start + 2, end):
                    for node_id in range(self.num_nodes):
                        n = node_id + (self.num_nodes * t) + robot_offset
                        if node_id == goal_node:
                            fixed[n] = 1
                        else:
                            fixed[n] = 0
            
            # self.logger.debug(f"Reachable positions for robot {robot_id}: {reachable_at_time}")
            # Fix unreachable nodes to 0 for all time steps for this robot
            for t in range(start + 1, end):
            # for t in reachable_at_time:
                for node_id in range(self.num_nodes):
                    var_idx = node_id + (self.num_nodes * t) + robot_offset
                    if node_id not in reachable_at_time.get(t, set([(goal_node)])):
                        fixed[var_idx] = 0

        return fixed

    def reachable_positions(self, robot, start_node, start, end):
        # Initialize reachable set with the start node at start_time
        reachable_at_time = {start: {start_node}}

        # Perform a BFS-like traversal to find all reachable nodes at each time step
        for t in range(start, end - 1):
            reachable_at_time[t + 1] = set()
            for node_i in reachable_at_time[t]:
                for node_j, _ in self.graph.adjacency.get(node_i, []):
                    reachable_at_time[t + 1].add(node_j)

        return reachable_at_time

    def reachable_positions_aggressive(self, robot, start_node, start_time, end_time):
        
        goal_node = self.problem.get_graph_robot_current_goal(robot.robot_id)[1]
        reachable_at_time = {start_time: {start_node}}
        visited = {start_node}  # Prevent revisiting previously reached nodes

        # Match QUBOBuilder loop range: start at start_time + 1, end at end_time
        for t in range(start_time + 1, end_time):
            prev_layer = reachable_at_time[t - 1]
            curr_layer = set()
            
            for node_i in prev_layer:
                for node_j, _ in self.graph.adjacency.get(node_i, []):
                    # Only expand to new nodes not yet visited
                    if node_j not in visited:
                        curr_layer.add(node_j)
                        visited.add(node_j)
            # Stop early if no new nodes are reachable
            if not curr_layer:
                break

            # To make sure goal is always reachable (and not conflict with goal lock)
            if goal_node in visited:
                curr_layer.add(goal_node)

            reachable_at_time[t] = curr_layer
        
        # print(reachable_at_time)
        return reachable_at_time

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

