import pennylane as qml


class QUBOBuilder:
    def __init__(self, problem, penalties, name="unnamed",
                 var_limit=176, window_max_steps=None):
        self.problem = problem
        self.penalties = penalties
        self.name = name  # Name for the penalties
        self.var_limit = var_limit  # Maximum number of variables in the QUBO
        self.t_max = window_max_steps or self.max_window_size()
        self.total_t = self.problem.T
        self.iter = 0
        self.T = min(self.t_max, self.total_t - (self.iter * self.t_max))
        self.initial_pos = problem.start  # Copy of the initial start position
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
        K_hot = self.penalties['K_hot']
        
        for t in range(self.T):
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
        adjacency = self.problem.grid.adjacency
        K_adj = self.penalties['K_adj']
        
        for t in range(self.T - 1):
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
        adjacency = self.problem.grid.adjacency
        K_adj = self.penalties['K_adj']
        
        for t in range(self.T - 1):
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
        K_goal = self.penalties['K_goal']

        # We start at time step 1 to not conflict with the start position
        for t in range(1, self.T):
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
        K_goal = self.penalties['K_goal']
        
        if self.iter == 0:
            K_goal = 0.4 * K_goal

        for t in range(1, self.T):
            goal_idx = e_i * N + e_j + M * N * t
            time_factor = 1 + (t / self.T)
            self.Q[(goal_idx, goal_idx)] += -K_goal * time_factor

    def apply_goal_early_penalty(self):
        """
        Apply early goal penalty: encourage reaching the goal earlier.
        It's stronger early and then decreases over time (till it reaches normal goal penalty).
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        K_goal = self.penalties['K_goal']

        for t in range(1, self.T):
            goal_idx = e_i * N + e_j + M * N * t
            time_factor = 1 + (self.T - t) / self.T
            self.Q[(goal_idx, goal_idx)] += -K_goal * time_factor

    def apply_lock_after_goal(self):
        """
        Apply lock after goal: discourage leaving the goal position once reached.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        K_lock = self.penalties['K_lock']

        for t in range(self.T - 1):  # up to T-2 to reference t+1
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
        K_bt = self.penalties['K_bt']
        for i in range(M):
            for j in range(N):
                # Skip goal — allow multiple visits
                if (i == e_i and j == e_j):
                    continue

                # For all time pairs t1 < t2
                for t1 in range(self.T):
                    g_t = i * N + j + M * N * t1
                    for t2 in range(t1 + 1, self.T):
                        g_t2 = i * N + j + M * N * t2
                        self.Q[(g_t, g_t2)] = self.Q.get((g_t, g_t2), 0) + K_bt

    def apply_tp_penalty(self):
        """
        Apply a penlaty if goal is reached before is even physically possible
        (i.e. if the goal is reached at time step t, but the manhattan distance is T)
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        min_steps = self.problem.manhattan_distance()
        
        K_tp = self.penalties['K_tp']

        for t in range(min(min_steps, self.T)):
            goal_idx = e_i * N + e_j + M * N * t
            self.Q[(goal_idx, goal_idx)] += K_tp  # Penalty for arriving to soon

    def apply_terrain_penalty(self):
        """
        Apply terrain penalty: encourage moving to cells with lower terrain cost.
        It introduces a linear bias depending on material costs in the grid.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        
        K_ter = self.penalties['K_ter']

        for t in range(self.T):  # t = 0 to T-1
            for i in range(M):
                for j in range(N):
                    material = self.problem.grid.get_terrain_at(i, j)
                    cost = self.problem.grid.get_material_cost(material)
                    # print(f"Applying terrain penalty at t={t}, i={i}, j={j}")
                    # print(f"Material: {material}, Cost: {cost}")
                    g_t = i * N + j + M * N * t
                    self.Q[(g_t, g_t)] += K_ter * cost

    def apply_elevation_penalty(self):
        """
        Apply elevation penalty: encourage moving to cells with lower elevation.
        This is similar to terrain penalty but focuses on elevation values.
        Be mind that if the slope is too steep, it will penalize the movement (for security reasons).
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        adjacency = self.problem.grid.adjacency
        K_elev = self.penalties['K_elev']

        for t in range(self.T - 1):
            for i in range(M):
                for j in range(N):
                    hi = self.problem.grid.get_elevation_at(i, j)
                    n = i * N + j + M * N * t  # linear index for time step t

                    for (k, l) in adjacency[(i, j)]:
                        hk = self.problem.grid.get_elevation_at(k, l)
                        delta_h = hk - hi  # positive = uphill

                        m = k * N + l + M * N * (t + 1)  # next time step index

                        if delta_h > 0:
                            move_cost = K_elev * (delta_h ** 1.8)  # super-linear for steep climbs
                        elif delta_h < -.7:
                            move_cost = K_elev * 0.7 * abs(delta_h)  # still costly if too steep down
                        else:
                            move_cost = K_elev * 0.3 * abs(delta_h)  # mild descent = easy

                        # Add to QUBO: only if move occurs
                        self.Q[(n, m)] = self.Q.get((n, m), 0) + move_cost

    def build(self, constraints_to_apply=None):
        if constraints_to_apply is None:
            penalty_to_constraint = {
                "K_hot": "one_hot",
                "K_adj": "adjacency_reward",
                "K_start": "start",
                "K_goal": "goal_later",
                "K_lock": "lock",
                "K_bt": "backtracking",
                "K_tp": "tp",
                "K_ter": "terrain",
                "K_elev": "elevation"
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
        if "tp" in constraints_to_apply:
            self.apply_tp_penalty()
        if "terrain" in constraints_to_apply:
            self.apply_terrain_penalty()
        if "elevation" in constraints_to_apply:
            self.apply_elevation_penalty()

        return self.Q
    
    def qubo_to_ising(self):
        """
        Convert the QUBO (upper triangle) dictionary to an Ising Hamiltonian format.
        Returns an qml.Hamiltonian object and an offset constant.
        """
        linear_coeffs = {}
        quadratic_terms = {}
        constant = 0.0
        qubit_indices = set()

        for (i, j), qij in self.Q.items():
            qubit_indices.update([i, j])
            if i == j:
                constant += qij / 2
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 2
            else:
                constant += qij / 4
                linear_coeffs[i] = linear_coeffs.get(i, 0) - qij / 4
                linear_coeffs[j] = linear_coeffs.get(j, 0) - qij / 4
                quadratic_terms[i, j] = quadratic_terms.get((i, j), 0) + qij / 4

        coeffs = []
        observables = []

        for i in sorted(linear_coeffs):
            if linear_coeffs[i] != 0:
                coeffs.append(linear_coeffs[i])
                observables.append(qml.PauliZ(i))

        for (i, j), val in quadratic_terms.items():
            if val != 0:
                coeffs.append(val)
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

        Hc = qml.Hamiltonian(coeffs, observables)
        return Hc, constant
    
    # def np_qubo_to_ising(self, Q):
    #     n = Q.shape[0]
    #     linear_coeffs = np.zeros(n)
    #     quadratic_terms = {}
    #     constant = 0.0

    #     for i in range(n):
    #         # Diagonal (linear) terms
    #         qii = Q[i, i]
    #         constant += qii / 2
    #         linear_coeffs[i] -= qii / 2  # from x_i = (1 - Z_i)/2

    #         for j in range(i+1, n):
    #             qij = Q[i, j]
    #             if qij != 0:
    #                 constant += qij / 4
    #                 linear_coeffs[i] -= qij / 4
    #                 linear_coeffs[j] -= qij / 4
    #                 quadratic_terms[(i, j)] = quadratic_terms.get((i, j), 0) + qij / 4

    #     coeffs = []
    #     observables = []

    #     for i in range(n):
    #         if linear_coeffs[i] != 0:
    #             coeffs.append(linear_coeffs[i])
    #             observables.append(qml.PauliZ(i))

    #     for (i, j), val in quadratic_terms.items():
    #         coeffs.append(val)
    #         observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

    #     H = qml.Hamiltonian(coeffs, observables)
    #     return H, constant
    
    def get_num_wires(self):
        """ 
        Get the number of qubits (wires) in the QUBO.
        It should return the number of unique qubit indices used in the QUBO dictionary.
        """
        # Standard way to retrieve out of the QUBO dictionary
        # if not self.Q:
        #     raise ValueError("QUBO dictionary is empty. Build the QUBO first.")
        # qubit_indices = set()
        # for (i, j) in self.Q.keys():
        #     qubit_indices.update([i, j])
        # return len(qubit_indices)

        # Fast way to extract it knowing the problem grid size and time steps
        return self.problem.grid.M * self.problem.grid.N * self.T

    def max_window_size(self):
        """
        Calculate the maximum window size based on the problem's time steps.
        It assumes current max is around 176+ variables so that will be the threshold
        """
        M = self.problem.grid.M
        N = self.problem.grid.N
        return self.var_limit // (M * N)

    def update_problem(self, new_start):
        """ It is meant to be used by the solver to update the problem after each iteration """
        self.iter += 1
        new_T = min(self.t_max, self.total_t - (self.iter * self.t_max))
        if (new_T > 0):
            self.problem.start = new_start 
            # print(f"Updated problem start to {self.problem.start} at iteration {self.iter}")
            self.T = new_T
            self.build()

    def reset_problem(self):
        """ It is meant to be used by the solver to reset the problem to the initial position """
        self.problem.start = self.initial_pos
        self.iter = 0
        self.T = min(self.t_max, self.total_t - (self.iter * self.t_max))
        self.build()
