from .base_qubo import BaseQUBO


class GraphQUBO(BaseQUBO):
    def __init__(
        self,
        problem,
        penalties,
        name="graph",
        var_limit=601,
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
                v for k, v in penalty_to_constraint.items() if k in self.penalties
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
            
        return self.Q

    def apply_one_hot(self):
        """Apply one-hot constraint: exactly one node per time step."""
        K_hot = self.penalties['K_hot']
        
        for t in range(self.T):
            # All node variables at time t
            # Formula: node_id + num_nodes * t
            indices = [node_id + (self.num_nodes * t) for node_id in range(self.num_nodes)]
            
            for n in indices:
                self.Q[(n, n)] = self.Q.get((n, n), 0) - K_hot
            
            for i, n in enumerate(indices):
                for m in indices[i + 1:]:
                    self.Q[(n, m)] = self.Q.get((n, m), 0) + 2 * K_hot

    def apply_start_penalty(self):
        """Apply start node penalty: must start at the given node."""
        K_start = self.penalties['K_start']
        start_node = self.problem.get_graph_start_end()[0]
        
        start_idx = start_node  # + (self.num_nodes * 0)
        self.Q[(start_idx, start_idx)] = (
            self.Q.get((start_idx, start_idx), 0) - K_start
        )

    def apply_goal_penalty(self):
        """Apply goal node penalty: encourage reaching the goal."""
        K_goal = self.penalties['K_goal']
        goal_node = self.problem.get_graph_start_end()[1]
        
        for t in range(1, self.T):
            goal_idx = goal_node + (self.num_nodes * t)
            time_factor = 1 + (t / self.T)
            self.Q[(goal_idx, goal_idx)] = (
                self.Q.get((goal_idx, goal_idx), 0) - K_goal * time_factor
            )

    def apply_lock_after_goal(self):
        """Apply lock-after-goal constraint: once at goal, stay there."""
        K_lock = self.penalties['K_lock']
        goal_node = self.problem.get_graph_start_end()[1]
        
        for t in range(self.T - 1):
            goal_idx_t = goal_node + (self.num_nodes * t)
            goal_idx_t1 = goal_node + (self.num_nodes * (t + 1))
            self.Q[(goal_idx_t, goal_idx_t)] = (
                self.Q.get((goal_idx_t, goal_idx_t), 0) + K_lock
            )
            self.Q[(goal_idx_t, goal_idx_t1)] = (
                self.Q.get((goal_idx_t, goal_idx_t1), 0) - K_lock
            )

    def apply_adjacency_constraint(self):
        """
        Apply adjacency constraint: only move between connected nodes.
        It enforces edge movements and penalizes non-adjacent moves.
        """
        K_adj = self.penalties['K_adj']
        adjacency = self.graph.adjacency
        
        for t in range(self.T - 1):
            for node_i in range(self.num_nodes):
                n = node_i + (self.num_nodes * t)
                # Skip if no connections
                # if node_i not in adjacency:
                #     continue

                # Reward staying at connected nodes
                for (node_j, weight) in adjacency[node_i]:
                    m = node_j + (self.num_nodes * (t + 1))
                    self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj * weight
                
                # Penalize moving to non-adjacent nodes
                for node_j in range(self.num_nodes):
                    if node_j != node_i and (node_j, 1.0) not in adjacency[node_i]:
                        m = node_j + (self.num_nodes * (t + 1))
                        self.Q[(n, m)] = self.Q.get((n, m), 0) + K_adj

    def apply_adjacency_reward(self):
        """Apply adjacency reward: encourage moving to adjacent nodes."""
        K_adj = self.penalties['K_adj']
        adjacency = self.graph.adjacency
        
        for t in range(self.T - 1):
            for node_i in range(self.num_nodes):
                n = node_i + (self.num_nodes * t)
                # Skip if no connections
                # if node_i not in adjacency:
                #     continue

                self.Q[(n, n)] = self.Q.get((n, n), 0) + K_adj

                # Reward staying at connected nodes
                for (node_j, weight) in adjacency[node_i]:
                    m = node_j + (self.num_nodes * (t + 1))
                    self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj * weight

    def apply_backtracking_penalty(self):
        """Apply backtracking penalty: discourage revisiting nodes."""
        K_bt = self.penalties['K_bt']
        goal_node = self.problem.get_graph_start_end()[1]
        
        for node_i in range(self.num_nodes):
            # Skip goal node - allow multiple visits
            if node_i == goal_node:
                continue
                
            # For all time pairs t1 < t2
            for t1 in range(self.T):
                n1 = node_i + (self.num_nodes * t1)
                for t2 in range(t1 + 1, self.T):
                    n2 = node_i + (self.num_nodes * t2)
                    self.Q[(n1, n2)] = self.Q.get((n1, n2), 0) + K_bt


    def get_fixed_variables(self):
        """Identify variables that can be fixed based on current problem state."""
        fixed = {}
        
        # Fix start node at time 0
        start_node = self.problem.get_graph_start_end()[0]
        start_idx = start_node
        fixed[start_idx] = 1
        
        # Fix all other nodes at time 0 to 0
        for node_i in range(self.num_nodes):
            if node_i != start_node:
                n = node_i + (self.num_nodes * 0)
                fixed[n] = 0

        # Initialize reachable set with the start node at time 0
        reachable_at_time = {0: {start_node}}

        # Perform a BFS-like traversal to find all reachable nodes at each time step
        for t in range(self.T - 1):
            reachable_at_time[t + 1] = set()
            for node_i in reachable_at_time[t]:
                for node_j, _ in self.graph.adjacency.get(node_i, []):
                    reachable_at_time[t + 1].add(node_j)

        # Fix unreachable nodes to 0 for all time steps
        # print("Reachable nodes at each time step:", reachable_at_time)
        for t in range(1, self.T):
            for node_id in range(self.num_nodes):
                var_idx = node_id + (self.num_nodes * t)
                if node_id not in reachable_at_time.get(t, set()):
                    fixed[var_idx] = 0

        return fixed

    def max_window_size(self):
        """Calculate max window size for graph-based problems."""
        return max(1, self.var_limit // self.num_nodes)
