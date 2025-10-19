import numpy as np
from .base_qubo import BaseQUBO


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


class GridQUBOBuilder(BaseQUBO):
    def __init__(
        self,
        problem,
        penalties,
        name="grid",
        var_limit=601,  # 605 10x10
        window_max_steps=None,
        distance_scaling="enhanced_linear",
    ):
        super().__init__(
            problem,
            penalties,
            name=name,
            var_limit=var_limit,
            window_max_steps=window_max_steps,
            distance_scaling=distance_scaling,
        )
        self.initial_num_vars = problem.grid.M * problem.grid.N * problem.num_robots * problem.T
        self.P_obs = compute_obstacle_potential_field(
            self.problem.grid.M,
            self.problem.grid.N,
            self.problem.grid.obstacles,
        )
        print("Window max steps:", self.max_window_size())

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
            K_dis = K_goal_approx * (1 / (1 + dist_to_goal)) * time_factor
        elif self.distance_scaling == "quadratic":
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
            K_dis = K_goal_approx * (1 / (1 + dist_to_goal)) * time_factor
        return K_dis

    # Must have exactly one position per time step
    def apply_one_hot(self):
        """
        Apply one-hot encoding constraint: exactly one position per time step.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_hot = self.penalties['K_hot']
        
        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            start_time = self.problem.robots[robot_id].start_time
            for t in range(start_time, self.T):
                indices = [i * N + j + (M * N) * t + robot_offset for i in range(M) for j in range(N)]
                
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
        
        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            start_time = self.problem.robots[robot_id].start_time
            for t in range(start_time, self.T - 1):
                for i in range(M):
                    for j in range(N):
                        # This a consistent linear indexing for the grid
                        n = i * N + j + M * N * t + robot_offset
                        self.Q[(n, n)] = self.Q.get((n, n), 0) + K_adj
                        
                        for (k, l) in adjacency[(i, j)]:
                            m = k * N + l + M * N * (t + 1) + robot_offset
                            self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj

    def apply_adjacency_penalty(self):
        """
        Apply adjacency penalty: discourage moving to non-adjacent cells.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        adjacency = self.problem.grid.adjacency
        K_adj = self.penalties['K_adj']
        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            start_time = self.problem.robots[robot_id].start_time
            for t in range(start_time, self.T - 1):
                for i in range(M):
                    for j in range(N):
                        n = i * N + j + M * N * t + robot_offset
                        
                        # Look at all possible positions at next time step
                        for k in range(M):
                            for l in range(N):
                                m = k*N + l + M*N*(t+1) + robot_offset

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
        K_start = self.penalties['K_start']

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            s_i, s_j = robot.current_position
            robot_time_start = robot.start_time
            robot_offset = robot_num * (M * N * self.T)
            start_idx = s_i * N + s_j + M * N * robot_time_start + robot_offset
            self.Q[(start_idx, start_idx)] = (
                self.Q.get((start_idx, start_idx), 0) - K_start
            )

    def apply_goal_approximation_penalty(self):
        """
        Apply goal approximation penalty: encourage getting near the goal.
        This is used when reaching goal is not possible in a single window.
        Uses a more balanced approach between goal attraction and obstacle 
        avoidance.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_goal_approx = self.penalties['K_goal_approx']
        K_obs_repel = 0.4
        
        # Apply to all robots
        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot = self.problem.robots[robot_id]
            e_i, e_j = robot.goal
            robot_time_start = robot.start_time
            robot_offset = robot_num * (M * N * self.T)
            
            for t in range(robot_time_start + 1, self.T):
                time_factor = (1.2) ** (5 * t / self.T)
                for i in range(M):
                    for j in range(N):
                        n = i * N + j + M * N * t + robot_offset
                        
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
                        self.Q[(n, n)] = (
                            self.Q.get((n, n), 0.0) - K_dis + K_obs
                        )
        # for t in range(1, self.T):
        #     time_factor = (1.2) ** (5 * t / self.T)
        #     for i in range(M):
        #         for j in range(N):
        #             for (i_next, j_next) in adjacency[(i, j)]:
        #                 n = i * N + j + M * N * t
        #                 n_next = i_next * N + j_next + M * N * (t+1)
        #                 delta = self.problem.manhattan_distance((i, j), (e_i, e_j)) - self.problem.manhattan_distance((i_next, j_next), (e_i, e_j))
        #                 reward = K_goal_approx * delta * time_factor
        #                 self.Q[(n, n_next)] = self.Q.get((n, n_next), 0.0) - reward


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
        K_goal = self.penalties['K_goal']
        if self.T + (self.iter * self.t_max) != self.total_t:
            self.apply_goal_approximation_penalty()
            # print("dx")
        else:
            for robot_num, robot_id in enumerate(self.problem.robots.keys()):
                robot = self.problem.robots[robot_id]
                e_i, e_j = robot.goal
                robot_time_start = robot.start_time
                robot_offset = robot_num * (M * N * self.T)
                for t in range(robot_time_start + 1, self.T):
                    goal_idx = e_i * N + e_j + M * N * t + robot_offset
                    time_factor = 1 + ((t - robot_time_start) / self.T)
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
        K_lock = self.penalties['K_lock']

        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            robot = self.problem.robots[robot_id]
            e_i, e_j = robot.goal
            start_time = robot.start_time
            for t in range(start_time, self.T - 1):  # up to T-2 to reference t+1
                g_t = e_i * N + e_j + M * N * t + robot_offset
                g_t_next = e_i * N + e_j + M * N * (t + 1) + robot_offset

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
        K_bt = self.penalties['K_bt']
        
        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            robot = self.problem.robots[robot_id]
            start_time = robot.start_time
            e_i, e_j = robot.goal  # Use robot's own goal

            for i in range(M):
                for j in range(N):
                    # Skip goal — allow multiple visits
                    if (i == e_i and j == e_j):
                        continue

                    # For all time pairs t1 < t2
                    for t1 in range(start_time, self.T):
                        g_t = i * N + j + M * N * t1 + robot_offset
                        for t2 in range(t1 + 1, self.T):
                            g_t2 = i * N + j + M * N * t2 + robot_offset
                            self.Q[(g_t, g_t2)] = (
                                self.Q.get((g_t, g_t2), 0) + K_bt
                            )

            if robot.active and robot.path:
                len_sol = len(robot.path)
                # print("robot path", robot.path)
                for t in range(start_time, self.T):
                    for p_idx, pos in enumerate(robot.path):
                        i, j = pos[0][:2]
                        idx = i * N + j + M * N * t + robot_offset
                        time_factor = (1 + (len_sol - p_idx)) / len_sol
                        self.Q[(idx, idx)] = self.Q.get((idx, idx), 0) + K_bt * time_factor

    def apply_tp_penalty(self):
        """
        Apply a penalty if goal is reached before is even physically possible
        (i.e. if the goal is reached at time step t, but the manhattan distance is T)
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_tp = self.penalties['K_tp']

        for robot_num, id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            robot = self.problem.robots[id]
            start_time = robot.start_time
            e_i, e_j = robot.goal
            min_steps = self.problem.manhattan_distance(
                robot.current_position, robot.goal
            )
            for t in range(start_time, min(min_steps + start_time, self.T)):
                goal_idx = e_i * N + e_j + M * N * t + robot_offset
                self.Q[(goal_idx, goal_idx)] += K_tp  # Penalty for arriving to soon

    def apply_terrain_penalty(self):
        """
        Apply terrain penalty: encourage moving to cells with lower terrain cost.
        It introduces a linear bias depending on material costs in the grid.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_ter = self.penalties['K_ter']
        for robot_num, id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            for t in range(self.T):
                for i in range(M):
                    for j in range(N):
                        material = self.problem.grid.get_terrain_at(i, j)
                        cost = self.problem.grid.get_material_cost(material)
                        # print(f"Applying terrain penalty at t={t}, i={i}, j={j}")
                        # print(f"Material: {material}, Cost: {cost}")
                        g_t = i * N + j + M * N * t + robot_offset
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

        for robot_num, id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            for t in range(self.T - 1):
                for i in range(M):
                    for j in range(N):
                        hi = self.problem.grid.get_elevation_at(i, j)
                        n = i * N + j + M * N * t + robot_offset  # linear index for time step t

                        for (k, l) in adjacency[(i, j)]:
                            hk = self.problem.grid.get_elevation_at(k, l)
                            delta_h = hk - hi  # positive = uphill

                            m = k * N + l + M * N * (t + 1) + robot_offset  # next time step index

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
        for robot_num, id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            for t in range(self.T):
                for (obs_i, obs_j) in self.problem.grid.obstacles:
                    obs_idx = obs_i * N + obs_j + M * N * t + robot_offset
                    self.Q[(obs_idx, obs_idx)] = (
                        self.Q.get((obs_idx, obs_idx), 0) + K_obs
                    )
    
    def apply_multi_robot_penalty(self):
        M, N = self.problem.grid.M, self.problem.grid.N
        K_crash = self.penalties.get('K_crash', 5)
        for t in range(self.T):
            for i in range(M):
                for j in range(N):
                    for r1 in range(num_robots):
                        for r2 in range(r1 + 1, num_robots):
                            idx1 = var_index_grid(r1, i, j, t)
                            idx2 = var_index_grid(r2, i, j, t)
                            self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_col

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

    def get_fixed_variables(self):
        M, N = self.problem.grid.M, self.problem.grid.N
        fixed = {}

        # Fix start position at time 0
        for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            robot_offset = robot_num * (M * N * self.T)
            robot = self.problem.robots[robot_id]
            s_i, s_j = robot.current_position
            start_time = robot.start_time
            start_idx = s_i * N + s_j + M * N * start_time + robot_offset

            fixed[start_idx] = 1
            print(start_idx, "fixed to 1 for robot", robot_id)
            # fix all other time=0 cells
            for i in range(M):
                for j in range(N):
                    n = i * N + j + M * N * start_time + robot_offset
                    if n != start_idx:
                        fixed[n] = 0

        # Fix goal position at last time step if within window
        # if (self.T + (self.iter * self.t_max)) == self.total_t:
        #     e_i, e_j = self.problem.end
        #     goal_idx = e_i * N + e_j + M * N * (self.T - 1)
        #     fixed[goal_idx] = 1

        # Fix obstacle cells to 0 at all time steps
        # for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            # robot_offset = robot_num * (M * N * self.T)
            # start_time = self.problem.robots[robot_id].start_time
            for t in range(start_time, self.T):
                for (obs_i, obs_j) in self.problem.grid.obstacles:
                    obs_idx = obs_i * N + obs_j + M * N * t + robot_offset
                    fixed[obs_idx] = 0

        # Now we can also fix unreachable cells to 0
        # Based on bfs (this essentially complies with adjacency and tp constraints)
        # for robot_num, robot_id in enumerate(self.problem.robots.keys()):
            reachable = self.reachable_mask(robot)
            for t in reachable:
                for i in range(M):
                    for j in range(N):
                        if not (i, j) in reachable[t]:
                            n = i * N + j + M * N * t + robot_offset
                            fixed[n] = 0

        return fixed

    def reachable_mask(self, robot):
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
        start = robot.current_position
        start_time = robot.start_time
        adjacency = self.problem.grid.adjacency  # Note that my adjacency map don't include obstacles
        obstacles = self.problem.grid.obstacles

        if obstacles is None:
            obstacles = []

        obstacles = set(obstacles)
        reachable = {start_time: {start}}

        for t in range(start_time + 1, T):
            prev_layer = reachable[t-1]
            curr_layer = set()

            for (i, j) in prev_layer:
                for (ni, nj) in adjacency.get((i, j), []):
                    if (ni, nj) not in obstacles:
                        curr_layer.add((ni, nj))

            reachable[t] = curr_layer

        return reachable


# Backward-compatible alias
QUBOBuilder = GridQUBOBuilder
