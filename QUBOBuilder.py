class QUBOBuilder:
    def __init__(self, problem, penalties):
        self.problem = problem
        self.penalties = penalties
        self.Q = {}

    # Must have exactly one position per time step
    def apply_one_hot(self):
        """
        Apply one-hot encoding constraint: exactly one position per time step.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        T = self.problem.T
        K_hot = self.penalties['K_hot']
        
        for t in range(T):
            indices = [i * M + j + (M * N) * t for i in range(M) for j in range(N)]
            
            for n in indices:
                self.Q[(n, n)] = self.Q.get((n, n), 0) - K_hot
            
            for i, n in enumerate(indices):
                for m in indices[i + 1:]:
                    self.Q[(n, m)] = self.Q.get((n, m), 0) + 2 * K_hot

    # Movement must be to adjacent cells
    # Be mind that all three approach need different constants to work well
    def apply_adjacency_reward(self):
        """
        Apply adjacency reward: encourage moving to adjacent cells.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        T = self.problem.T
        adjacency = self.problem.grid.adjacency
        K_adj = self.penalties['K_adj']
        
        for t in range(T - 1):
            for i in range(M):
                for j in range(N):
                    # This a consistent linear indexing for the grid
                    n = i * N + j + M * N * t
                    self.Q[(n, n)] = self.Q.get((n, n), 0) + K_adj
                    
                    for (k, l) in adjacency[(i, j)]:
                        m = k * N + l + M * N * (t + 1)
                        self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj

    def apply_adjacency_penalty(self):
        """
        Apply adjacency penalty: discourage moving to non-adjacent cells.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        T = self.problem.T
        adjacency = self.problem.grid.adjacency
        K_adj = self.penalties['K_adj']
        
        for t in range(T - 1):
            for i in range(M):
                for j in range(N):
                    n = i * N + j + M * N * t
                    
                    # Look at all possible positions at next time step
                    for k in range(M):
                        for l in range(N):
                            m = k*N + l + M*N*(t+1)

                            # Skip self-loop unless explicitly allowed
                            if (k == i and l == j):
                                continue

                            # Only penalize if (k,l) is NOT in adjacency[(i,j)]
                            if (k, l) not in adjacency[(i, j)]:
                                self.Q[(n, m)] = self.Q.get((n, m), 0) + K_adj

    def apply_start_penalty(self):
        """
        Apply start position penalty: must start at the given position.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        s_i, s_j = self.problem.start
        K_start = self.penalties['K_start']
        
        start_idx = s_i * N + s_j + M * N * 0  # Time step 0
        self.Q[(start_idx, start_idx)] += -K_start

    def apply_goal_fix_penalty(self):
        """
        Apply goal position penalty: encourage reaching the goal.
        This is equally applied at all time steps after the start.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        T = self.problem.T
        K_goal = self.penalties['K_goal']
        
        # We start at time step 1 to not conflict with the start position
        for t in range(1, T):
            goal_idx = e_i * N + e_j + M * N * t
            self.Q[(goal_idx, goal_idx)] += -K_goal

    def apply_goal_early_penalty(self):
        """
        Apply early goal penalty: encourage reaching the goal earlier.
        It's stronger early and then decreases over time (till it reaches normal goal penalty).
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        T = self.problem.T
        K_goal = self.penalties['K_goal']
        
        for t in range(1, T):
            goal_idx = e_i * N + e_j + M * N * t
            time_factor = 1 + (T - t) / T
            self.Q[(goal_idx, goal_idx)] += -K_goal * time_factor

    def apply_lock_after_goal(self):
        """
        Apply lock after goal: discourage leaving the goal position once reached.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        T = self.problem.T
        K_lock = self.penalties['K_lock']
        
        for t in range(T - 1):  # up to T-2 to reference t+1
            g_t = e_i * N + e_j + M * N * t
            g_t_next = e_i * N + e_j + M * N * (t + 1)

            # Linear term: +K_lock * x_g_t
            self.Q[(g_t, g_t)] = self.Q.get((g_t, g_t), 0) + K_lock

            # Quadratic term: -K_lock * x_g_t * x_g_t_next
            self.Q[(g_t, g_t_next)] = self.Q.get((g_t, g_t_next), 0) - K_lock

    def build(self, constraints_to_apply=None):
        constraints_to_apply = constraints_to_apply or [
            "one_hot", "adjacency_reward", "start", "goal_fix", "lock"
        ]

        if "one_hot" in constraints_to_apply:
            self.apply_one_hot()
        if "adjacency_reward" in constraints_to_apply:
            self.apply_adjacency_reward()
        if "adjacency_penalty" in constraints_to_apply:
            self.apply_adjacency_penalty()
        if "start" in constraints_to_apply:
            self.apply_start_penalty()
        if "goal_fix" in constraints_to_apply:
            self.apply_goal_fix_penalty()
        if "goal_early" in constraints_to_apply:
            self.apply_goal_early_penalty()
        if "lock" in constraints_to_apply:
            self.apply_lock_after_goal()

        return self.Q
