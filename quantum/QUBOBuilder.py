import pennylane as qml
import numpy as np


def compute_obstacle_potential_field(M, N, obstacles, sigma=1.5):
    """
    Compute a 2D potential field where each obstacle contributes
    a Gaussian bump. Sum over all obstacles.
    """
    P = np.zeros((M, N))
    for (oi, oj) in obstacles:
        # Create grid of distances
        ii, jj = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
        dist_sq = (ii - oi)**2 + (jj - oj)**2
        P += np.exp(-dist_sq / (2 * sigma**2))
    return P


class QUBOBuilder:
    def __init__(self, problem, penalties, name="unnamed",
                 var_limit=601, window_max_steps=None, distance_scaling="enhanced_linear"):
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
        self.distance_scaling = distance_scaling  # Method for Manhattan distance scaling
        self.P_obs = compute_obstacle_potential_field(
            self.problem.grid.M,
            self.problem.grid.N,
            self.problem.grid.obstacles
        )
        # self.result

    # Since the config file already returns a dict structure for the penalties, there is no need
    # to define a new dictionary constructor

    def calculate_manhattan_penalty(self, raw_dist, K_goal_approx, time_factor):
        """
        Calculate Manhattan distance penalty using the specified scaling method.
        
        Args:
            raw_dist: Raw Manhattan distance
            K_goal_approx: Goal approximation penalty coefficient
            time_factor: Time-based scaling factor
            
        Returns:
            K_dis: Calculated distance penalty
        """
        if self.distance_scaling == "enhanced_linear":
            # Enhanced linear scaling for small grids
            dist_to_goal = raw_dist * 0.165
            K_dis = K_goal_approx * (1/(0.7 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "exponential":
            # Exponential scaling for medium grids
            dist_to_goal = raw_dist * 1.2
            K_dis = K_goal_approx * (1/(1 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "quadratic":
            # Quadratic scaling for large grids
            dist_to_goal = raw_dist ** 1.3
            K_dis = K_goal_approx * (1/(1 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "logarithmic":
            # Logarithmic scaling for balanced approach
            dist_to_goal = np.log(1 + raw_dist * 2)
            K_dis = K_goal_approx * (1/(1 + dist_to_goal)) * time_factor
            
        elif self.distance_scaling == "adaptive":
            # Grid-size adaptive scaling
            grid_size = max(self.problem.grid.M, self.problem.grid.N)
            if grid_size <= 3:
                dist_to_goal = raw_dist * 0.4
                K_dis = K_goal_approx * (1/(0.2 + dist_to_goal)) * time_factor
            elif grid_size <= 5:
                dist_to_goal = raw_dist * 0.8
                K_dis = K_goal_approx * (1/(0.4 + dist_to_goal)) * time_factor
            else:
                dist_to_goal = raw_dist * 1.2
                K_dis = K_goal_approx * (1/(0.8 + dist_to_goal)) * time_factor
                
        else:  # Default to original formula
            dist_to_goal = raw_dist * 2
            K_dis = K_goal_approx * (1/(1 + dist_to_goal)) * time_factor
            
        return K_dis

    # Must have exactly one position per time step
    def apply_one_hot(self):
        """
        Apply one-hot encoding constraint: exactly one position per time step.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_hot = self.penalties['K_hot']
        
        for t in range(self.T):
            indices = [i * N + j + (M * N) * t for i in range(M) for j in range(N)]
            
            for n in indices:
                self.Q[(n, n)] = self.Q.get((n, n), 0) - K_hot
            
            for i, n in enumerate(indices):
                for m in indices[i + 1:]:
                    self.Q[(n, m)] = self.Q.get((n, m), 0) + 2 * K_hot

    # Be mind the two approaches need different constants to work well
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
        self.Q[(start_idx, start_idx)] = self.Q.get((start_idx, start_idx), 0) - K_start

    def apply_goal_approximation_penalty(self):
        """
        Apply goal approximation penalty: encourage getting near the goal.
        This is used when reaching goal is not possible in a single window.
        Uses a more balanced approach between goal attraction and obstacle 
        avoidance.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        K_goal_approx = self.penalties['K_goal_approx']
        K_obs_repel = 0.4
        
        for t in range(1, self.T):
            time_factor = (1.2)**(5 * t / self.T)
            for i in range(M):
                for j in range(N):
                    n = i * N + j + M * N * t
                    
                    # Skip obstacle cells entirely - let explicit obstacle 
                    # constraint handle them
                    if (i, j) in self.problem.grid.obstacles:
                        continue
                    
                    # Goal progress (Manhattan, time-weighted)
                    # Use the configurable Manhattan distance scaling method
                    raw_dist = self.problem.manhattan_distance((i, j), (e_i, e_j))
                    K_dis = self.calculate_manhattan_penalty(raw_dist, K_goal_approx, time_factor)

                    # Soft obstacle avoidance using potential field 
                    # (for nearby obstacles)

                    K_obs = K_obs_repel * self.P_obs[i, j]

                    self.Q[(n, n)] = self.Q.get((n, n), 0.0) - K_dis + K_obs

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

        if self.T + (self.iter * self.t_max) != self.total_t:
            self.apply_goal_approximation_penalty()

        else:
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
        min_steps = self.problem.manhattan_distance(self.problem.start, self.problem.end)
        
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

    def apply_obstacle_penalty(self):
        """
        Apply hard obstacle penalty: strongly discourage or prohibit entering 
        obstacle cells. This creates a much stronger constraint than the soft 
        potential field approach.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_obs = self.penalties.get('K_obs', 2)
        
        for t in range(self.T):
            for (obs_i, obs_j) in self.problem.grid.obstacles:
                # Add very high penalty for being in an obstacle cell
                obs_idx = obs_i * N + obs_j + M * N * t
                self.Q[(obs_idx, obs_idx)] = self.Q.get((obs_idx, obs_idx), 0) + K_obs

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
                "K_elev": "elevation",
                "K_obs": "obstacle"
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
        if "obstacle" in constraints_to_apply:
            self.apply_obstacle_penalty()

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
        if not self.Q:
            raise ValueError("QUBO dictionary is empty. Build the QUBO first.")
        qubit_indices = set()
        for (i, j) in self.Q.keys():
            qubit_indices.update([i, j])
        return len(qubit_indices)

        # Fast way to extract it knowing the problem grid size and time steps
        # return self.problem.grid.M * self.problem.grid.N * self.T

    def max_window_size(self):
        """
        Calculate the maximum window size based on the problem's time steps
        and the var_limit variable.
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
            self.T = new_T
            self.build()

    def reset_problem(self):
        """ It is meant to be used by the solver to reset the problem to the initial position """
        self.problem.start = self.initial_pos
        self.iter = 0
        self.T = min(self.t_max, self.total_t - (self.iter * self.t_max))
        # self.build()

    def dict_to_array(self, fill_value=0):
        if not self.Q:
            return np.array([[]])  # handle empty dict
        rows, cols = zip(*self.Q.keys())
        shape = (max(rows)+1, max(cols)+1)
        arr = np.full(shape, fill_value, dtype=float)
        for (r, c), val in self.Q.items():
            arr[r, c] = val
        return arr

    def reduce_qubo(self, fixed_vars):
        """
        Reduce a QUBO dictionary by fixing variables.

        Q: dict {(i,j): coeff} original QUBO
        fixed_vars can be:
        - dict {idx: value}
        - numpy array of length total_vars, with 0/1 for fixed vars and np.nan for free ones

        Returns:
            reduced_Q: dict with fixed variables removed and contributions applied
            const_offset: total constant offset added from variables fixed to 1
        """
        # convert np array -> dict if needed
        if isinstance(fixed_vars, np.ndarray):
            fixed_dict = {i: int(v) for i, v in enumerate(fixed_vars) if not np.isnan(v)}
        else:
            fixed_dict = fixed_vars
        
        Q = self.Q.copy()
        const_offset = 0

        for var, val in fixed_dict.items():
            if val == 0:
                # variable fixed to 0 → remove all terms involving var
                keys_to_remove = [k for k in Q if var in k]
                for k in keys_to_remove:
                    Q.pop(k, None)

            elif val == 1:
                # variable fixed to 1 → add contributions to neighbors
                neighbors = [j for (i, j) in Q if i == var and j != var] + \
                            [i for (i, j) in Q if j == var and i != var]

                for nb in neighbors:
                    coeff = Q.get((var, nb), 0) + Q.get((nb, var), 0)
                    Q[(nb, nb)] = Q.get((nb, nb), 0) + coeff

                # add diagonal term to constant offset
                const_offset += Q.get((var, var), 0)

                # remove all terms involving var
                keys_to_remove = [k for k in Q if var in k]
                for k in keys_to_remove:
                    Q.pop(k, None)

        return Q, const_offset


    def list_to_dict_solution(self, solution_list):
        """
        Convert list solution to dictionary format.
        
        solution_list: list of dicts [{idx: value, ...}] from solver
        
        Returns:
            dict {idx: np.int8(value)} 
        """
        if isinstance(solution_list, list) and len(solution_list) > 0:
            # Extract the dictionary from the list
            solution_dict = solution_list[0] if isinstance(solution_list[0], dict) else {}
        elif isinstance(solution_list, dict):
            solution_dict = solution_list
        else:
            solution_dict = {}
        
        return solution_dict


    def reconstruct_solution(self, reduced_sol, fixed_vars, total_vars):
        """
        Merge solver output with fixed variables into full solution dictionary.

        reduced_sol: dict {original_index: value} from solver OR list containing dict
        fixed_vars: dict {original_index: value} fixed before reduction
        total_vars: int, total number of variables in original QUBO
        
        Returns:
            dict {idx: np.int8(value)} in sequential key order (0, 1, 2, ...)
        """
        # Convert list to dict if needed
        if isinstance(reduced_sol, list):
            reduced_sol = self.list_to_dict_solution(reduced_sol)
        
        # Create full solution dictionary in sequential order
        full_sol_dict = {}
        
        # Build solution in order 0, 1, 2, ..., total_vars-1
        for i in range(total_vars):
            if i in fixed_vars:
                # Use fixed value
                full_sol_dict[i] = np.int8(fixed_vars[i])
            elif i in reduced_sol:
                # Use solver value
                full_sol_dict[i] = np.int8(reduced_sol[i])
            else:
                # Default to 0 for any missing variables
                full_sol_dict[i] = np.int8(0)

        return full_sol_dict
    
    def get_fixed_variables(self):
        """
        Identify variables that can be fixed based on current problem state.
        Returns a dict {idx: value} of variables to fix.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        fixed = {}

        # Fix start position at time 0
        s_i, s_j = self.problem.start
        start_idx = s_i * N + s_j
        fixed[start_idx] = 1
        # fix all other time=0 cells
        for i in range(M):
            for j in range(N):
                n = i * N + j
                if n != start_idx:
                    fixed[n] = 0

        # Fix goal position at last time step if within window
        # if (self.T + (self.iter * self.t_max)) == self.total_t:
        #     e_i, e_j = self.problem.end
        #     goal_idx = e_i * N + e_j + M * N * (self.T - 1)
        #     fixed[goal_idx] = 1

        # Fix obstacle cells to 0 at all time steps
        for t in range(self.T):
            for (obs_i, obs_j) in self.problem.grid.obstacles:
                obs_idx = obs_i * N + obs_j + M * N * t
                fixed[obs_idx] = 0

        # Now we can also fix unreachable cells to 0
        # Based on bfs (this essentially complies with adjacency and tp constraints)
        reachable = self.reachable_mask()
        for t in reachable:
            for i in range(M):
                for j in range(N):
                    if not (i, j) in reachable[t]:
                        n = i * N + j + M * N * t
                        fixed[n] = 0

        return fixed

    def reachable_mask(self):
        """
        Compute reachable positions per time step.
        Note that it is aggresive, since this one is based on my adjacency map, which does not include staying in place.
        This is the ideal scenario of always keep moving, but when implementing dynamic obstacles, we may want to consider staying in place as well.

        Args:
            T: number of time steps
            start: (i, j) tuple for start position
            adjacency: dict mapping (i,j) -> list of neighbor (i,j)
            obstacles: list of blocked (i,j), optional

        Returns:
            A dict: {t: set((i, j), ...)} of reachable positions per time step.
        """

        T = self.T
        start = self.problem.start
        adjacency = self.problem.grid.adjacency  # Note that my adjacency map don't include obstacles
        obstacles = self.problem.grid.obstacles

        if obstacles is None:
            obstacles = []

        obstacles = set(obstacles)
        reachable = {0: {start}}

        for t in range(1, T):
            prev_layer = reachable[t-1]
            curr_layer = set()

            for (i, j) in prev_layer:
                for (ni, nj) in adjacency.get((i, j), []):
                    if (ni, nj) not in obstacles:
                        curr_layer.add((ni, nj))

            reachable[t] = curr_layer

        return reachable

