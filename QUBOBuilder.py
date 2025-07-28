class QUBOBuilder:
    def __init__(self, problem, penalties, name="unnamed", var_limit=176, window_max_steps=None):
        self.problem = problem
        self.penalties = penalties
        self.name = name  # Name for the penalties
        self.var_limit = var_limit  # Maximum number of variables in the QUBO
        self.window_max_steps = window_max_steps or self.max_window_size()
        self.iter = 0
        self.Q = {}
        # self.result

    # Since the config file already returns a dict structure for the penalties, there is no need
    # to define a new dictionary constructor

    # Must have exactly one position per time step
    def apply_one_hot(self):
        """
        Apply one-hot encoding constraint: exactly one position per time step.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        T = self.problem.T
        # T = min(max_T, self.window_max_steps * (self.iter+1))  # Use the max window size if defined
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

    def apply_goal_later_penalty(self):
        """
        At each iteration it increases the penalty for not reaching the goal.
        This helps to avoid getting stuck in local minima. And found goal at later time
        Without forcing a teleportation at start position.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        T = self.problem.T
        K_goal = self.penalties['K_goal']
        
        for t in range(1, T):
            goal_idx = e_i * N + e_j + M * N * t
            time_factor = 1 + (t / T)
            self.Q[(goal_idx, goal_idx)] += -K_goal * time_factor

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

    def apply_backtracking_penalty(self):
        """
        Apply backtracking penalty: discourage moving back to the previous position.
        """
        # Constraint: No backtracking (except at goal)
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        T = self.problem.T
        K_bt = self.penalties['K_bt']
        for i in range(M):
            for j in range(N):
                # Skip goal — allow multiple visits
                if (i == e_i and j == e_j):
                    continue

                # For all time pairs t1 < t2
                for t1 in range(T):
                    g_t = i * N + j + M * N * t1
                    for t2 in range(t1 + 1, T):
                        g_t2 = i * N + j + M * N * t2
                        self.Q[(g_t, g_t2)] = self.Q.get((g_t, g_t2), 0) + K_bt

    def build(self, constraints_to_apply=None):
        if constraints_to_apply is None:
            penalty_to_constraint = {
                "K_hot": "one_hot",
                "K_adj": "adjacency_reward",
                "K_start": "start",
                "K_goal": "goal_later",
                "K_lock": "lock",
                "K_bt": "backtracking"
            }
            constraints_to_apply = [
                v for k, v in penalty_to_constraint.items()
                if k in self.penalties
            ]

        # To clean the QUBO dictionary before building
        # In case there were previous qubo with different constraints/size
        self.Q = {}

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
        if "goal_later" in constraints_to_apply:
            self.apply_goal_later_penalty()
        if "lock" in constraints_to_apply:
            self.apply_lock_after_goal()
        if "backtracking" in constraints_to_apply:
            self.apply_backtracking_penalty()

        return self.Q
    
    def max_window_size(self):
        """
        Calculate the maximum window size based on the problem's time steps.
        It assumes current max is around 176+ variables so that will be the threshold
        """
        M = self.problem.grid.M
        N = self.problem.grid.N
        return self.var_limit // (M * N)
