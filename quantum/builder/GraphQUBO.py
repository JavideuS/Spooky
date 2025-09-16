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
                "K_start": "start",
                "K_goal": "goal",
                "K_adj": "adjacency",
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
        if "backtracking" in constraints_to_apply:
            self.apply_backtracking_penalty()
            
        return self.Q

    def apply_one_hot(self):
        """Apply one-hot constraint: exactly one node per time step."""
        K_hot = self.penalties['K_hot']
        
        for t in range(self.T):
            # All node variables at time t
            indices = [node_id * self.T + t for node_id in range(self.num_nodes)]
            
            for n in indices:
                self.Q[(n, n)] = self.Q.get((n, n), 0) - K_hot
            
            for i, n in enumerate(indices):
                for m in indices[i + 1:]:
                    self.Q[(n, m)] = self.Q.get((n, m), 0) + 2 * K_hot

    def apply_start_penalty(self):
        """Apply start node penalty: must start at the given node."""
        K_start = self.penalties['K_start']
        start_node = self.problem.start
        
        start_idx = start_node * self.T + 0  # Time step 0
        self.Q[(start_idx, start_idx)] = (
            self.Q.get((start_idx, start_idx), 0) - K_start
        )

    def apply_goal_penalty(self):
        """Apply goal node penalty: encourage reaching the goal."""
        K_goal = self.penalties['K_goal']
        goal_node = self.problem.end
        
        for t in range(1, self.T):
            goal_idx = goal_node * self.T + t
            time_factor = 1.0 + (self.T - t) / self.T
            self.Q[(goal_idx, goal_idx)] = (
                self.Q.get((goal_idx, goal_idx), 0) - K_goal * time_factor
            )

    def apply_adjacency_constraint(self):
        """Apply adjacency constraint: only move between connected nodes."""
        K_adj = self.penalties['K_adj']
        adjacency = self.graph.adjacency
        
        for t in range(self.T - 1):
            for node_i in range(self.num_nodes):
                n = node_i * self.T + t
                
                # Reward staying at connected nodes
                for (node_j, weight) in adjacency[node_i]:
                    m = node_j * self.T + (t + 1)
                    self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj * weight
                
                # Penalize moving to non-adjacent nodes
                for node_j in range(self.num_nodes):
                    if node_j != node_i and (node_j, 1.0) not in adjacency[node_i]:
                        m = node_j * self.T + (t + 1)
                        self.Q[(n, m)] = self.Q.get((n, m), 0) + K_adj

    def apply_backtracking_penalty(self):
        """Apply backtracking penalty: discourage revisiting nodes."""
        K_bt = self.penalties['K_bt']
        goal_node = self.problem.end
        
        for node_i in range(self.num_nodes):
            # Skip goal node - allow multiple visits
            if node_i == goal_node:
                continue
                
            # For all time pairs t1 < t2
            for t1 in range(self.T):
                n1 = node_i * self.T + t1
                for t2 in range(t1 + 1, self.T):
                    n2 = node_i * self.T + t2
                    self.Q[(n1, n2)] = self.Q.get((n1, n2), 0) + K_bt

    def get_fixed_variables(self):
        """Identify variables that can be fixed based on current problem state."""
        fixed = {}
        
        # Fix start node at time 0
        start_node = self.problem.start
        start_idx = start_node * self.T + 0
        fixed[start_idx] = 1
        
        # Fix all other nodes at time 0 to 0
        for node_i in range(self.num_nodes):
            if node_i != start_node:
                n = node_i * self.T + 0
                fixed[n] = 0
        
        return fixed

    def max_window_size(self):
        """Calculate max window size for graph-based problems."""
        return max(1, self.var_limit // self.num_nodes)
