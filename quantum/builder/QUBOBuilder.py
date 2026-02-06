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
        var_limit=1650,  # 101 605 1001
        window_max_steps=None,
        distance_scaling="enhanced_linear",
        robot_window_limits=None,
        verbose_level=2,
    ):
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
        self.initial_num_vars = problem.grid.M * problem.grid.N * problem.num_robots * problem.T
        self.P_obs = compute_obstacle_potential_field(
            self.problem.grid.M,
            self.problem.grid.N,
            self.problem.grid.obstacles,
        )
        self.logger.standard("Window max steps:", self.max_window_size())

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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]
            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            # Note that if current_T > start_time then it is a continuation
            end_time = robot.T + start_time
            end = end_time - self.current_T
            # This would mean it doesn't finish in this window
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            for t in range(start, end):
                # print(t)
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)

            robot = self.problem.robots[robot_id]
            s_i, s_j = robot.current_position
            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T

            start_idx = s_i * N + s_j + M * N * start + robot_offset
            self.Q[(start_idx, start_idx)] = (
                self.Q.get((start_idx, start_idx), 0) - K_start
            )

    def apply_goal_approximation_penalty(self, robot_id):
        """
        Apply goal approximation penalty: encourage getting near the goal.
        This is used when reaching goal is not possible in a single window.
        Uses a more balanced approach between goal attraction and obstacle
        avoidance.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_goal_approx = self.penalties['K_goal_approx']
        K_obs_repel = 0.4
        robot_nums = self.problem.get_robot_nums()

        robot_offset = robot_nums[robot_id] * (M * N * self.total_t)

        robot = self.problem.robots[robot_id]
        e_i, e_j = robot.goal

        start_time = robot.start_time
        start = 0
        if self.current_T < start_time:
            start = start_time - self.current_T

        for t in range(start + 1, self.t_max):
            time_factor = (1.2) ** (5 * (t - start) / (self.t_max - start))
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot = self.problem.robots[robot_id]
            start_time = robot.start_time
            end_time = robot.T + start_time

            if end_time > self.current_T + self.t_max:
                # Had to make it single robot, else it would conflict with no approximation robots
                self.apply_goal_approximation_penalty(robot_id)
            else:
                robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
                e_i, e_j = robot.goal

                start = 0
                if self.current_T < start_time:
                    start = start_time - self.current_T

                end = end_time - self.current_T
                # This is used because in immediate goals and big windows the goal is not strong enough
                window_constant = 1 + (end / 10)
                # print("hola", robot_id)
                for t in range(start + 1, end):
                    goal_idx = e_i * N + e_j + M * N * t + robot_offset
                    # Note that since I initially considered all this time factor for single robot starting in t=-
                    # I adjust to keep the same growth by reducing time start to both)
                    time_factor = 1 + ((t - start) / (end - start))
                    self.Q[(goal_idx, goal_idx)] += -K_goal * time_factor * window_constant

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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]
            e_i, e_j = robot.goal

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            for t in range(start, end - 1):  # up to T-2 to reference t+1
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            e_i, e_j = robot.goal  # Use robot's own goal

            for i in range(M):
                for j in range(N):
                    # Skip goal — allow multiple visits
                    if (i == e_i and j == e_j):
                        continue

                    # For all time pairs t1 < t2
                    for t1 in range(start, end):
                        g_t = i * N + j + M * N * t1 + robot_offset
                        for t2 in range(t1 + 1, end):
                            g_t2 = i * N + j + M * N * t2 + robot_offset
                            self.Q[(g_t, g_t2)] = (
                                self.Q.get((g_t, g_t2), 0) + K_bt
                            )

            if robot.active and robot.path:
                len_sol = len(robot.path)
                # print("robot path", robot.path)
                for t in range(start, end):
                    for p_idx, pos in enumerate(robot.path):
                        # print(p_idx, pos)
                        i, j = pos[:2]
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            e_i, e_j = robot.goal
            min_steps = self.problem.manhattan_distance(
                robot.current_position, robot.goal
            )

            for t in range(start, min(min_steps + start, end)):
                goal_idx = e_i * N + e_j + M * N * t + robot_offset
                self.Q[(goal_idx, goal_idx)] += K_tp  # Penalty for arriving to soon

    def apply_terrain_penalty(self):
        """
        Apply terrain penalty: encourage moving to cells with lower terrain cost.
        It introduces a linear bias depending on material costs in the grid.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_ter = self.penalties['K_ter']
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            for t in range(start, end):
                for i in range(M):
                    for j in range(N):
                        material = self.problem.grid.get_terrain_at(i, j)
                        cost = self.problem.grid.get_material_cost(material)
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
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
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            for t in range(start, end):
                for (obs_i, obs_j) in self.problem.grid.obstacles:
                    obs_idx = obs_i * N + obs_j + M * N * t + robot_offset
                    self.Q[(obs_idx, obs_idx)] = (
                        self.Q.get((obs_idx, obs_idx), 0) + K_obs
                    )

    def apply_multi_robot_penalty(self):
        M, N = self.problem.grid.M, self.problem.grid.N
        K_crash = self.penalties.get('K_crash', 0)
        robot_nums = self.problem.get_robot_nums()
        active_robots_per_timestep = self.get_active_robots_per_timestep_in_window()
        for t, active_robots in active_robots_per_timestep.items():
            if len(active_robots) < 2:
                continue  # no collision possible
            for i in range(M):
                for j in range(N):
                    for robot_id1 in active_robots:
                        for robot_id2 in active_robots:
                            if robot_nums[robot_id1] >= robot_nums[robot_id2]:
                                continue

                            robot_offset1 = robot_nums[robot_id1] * (M * N * self.total_t)
                            robot_offset2 = robot_nums[robot_id2] * (M * N * self.total_t)
                            # Note that if I don't substract current_t it doesn't keep relative window time
                            idx1 = i * N + j + M * N * (t - self.current_T) + robot_offset1
                            idx2 = i * N + j + M * N * (t - self.current_T) + robot_offset2
                            self.Q[(idx1, idx2)] = self.Q.get((idx1, idx2), 0) + K_crash


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
                "K_obs": "obstacle",
                "K_crash": "multi_robot",
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
        if "multi_robot" in constraints_to_apply:
            self.apply_multi_robot_penalty()

        return self.Q

    def reachable_positions(self, robot, start_time, end_time):
        """
        Compute reachable positions per time step.
        Note that it is aggresive, since this one is based on my adjacency map, which does not include staying in place.
        This is the ideal scenario of always keep moving, but when implementing dynamic obstacles, we may want to consider staying in place as well.

        Args:


        Returns:
            A dict: {t: set((i, j), ...)} of reachable positions per time step.
        """

        start = robot.current_position
        adjacency = self.problem.grid.adjacency  # Note that my adjacency map don't include obstacles
        obstacles = self.problem.grid.obstacles

        if obstacles is None:
            obstacles = []

        obstacles = set(obstacles)
        reachable = {start_time: {start}}

        for t in range(start_time + 1, end_time):
            prev_layer = reachable[t-1]
            curr_layer = set()

            for (i, j) in prev_layer:
                for (ni, nj) in adjacency.get((i, j), []):
                    if (ni, nj) not in obstacles:
                        curr_layer.add((ni, nj))

            reachable[t] = curr_layer

        return reachable

    def reachable_positions_aggressive(self, robot,start, start_time, end_time):
        """
        Compute reachable positions per time step without backtracking.
        That means once a cell is reached, it won't be revisited in future time steps.

        Args:
            robot: Robot object with current_position.
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """

        goal = robot.goal
        adjacency = self.problem.grid.adjacency  # adjacency map without obstacles
        obstacles = set(self.problem.grid.obstacles or [])

        reachable = {start_time: {start}}
        visited = {start}  # <- prevent backtracking / revisiting

        # Expand layer by layer
        for t in range(start_time + 1, end_time):
            prev_layer = reachable[t - 1]
            curr_layer = set()

            for (i, j) in prev_layer:
                for (ni, nj) in adjacency.get((i, j), []):
                    if (ni, nj) not in obstacles and (ni, nj) not in visited:
                        curr_layer.add((ni, nj))
                        visited.add((ni, nj))  # mark as seen globally

            # If no new cells are reachable, you can stop early
            if not curr_layer:
                break

            # To make sure goal is always reachable (and not conflict with goal lock)
            if goal in visited:
                curr_layer.add(goal)

            reachable[t] = curr_layer

        return reachable

    def reachable_positions_aggressive_v2(self, robot, start, start_time, end_time):
        """
        Compute reachable positions per time step without backtracking,
        taking into account the robot's historical path (similar to backtracking penalty).

        This is even more aggressive than reachable_positions_aggressive because it also
        excludes positions that are in the robot's path from being visited again.

        Args:
            robot: Robot object with current_position and path attribute.
            start: Starting position tuple (i, j).
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """

        goal = robot.goal
        adjacency = self.problem.grid.adjacency  # adjacency map without obstacles
        obstacles = set(self.problem.grid.obstacles or [])

        reachable = {start_time: {start}}
        visited = {start}  # <- prevent backtracking / revisiting

        # Add robot's historical path to visited set (excluding goal to allow re-entry)
        if robot.active and robot.path:
            for pos in robot.path:
                # Extract (i, j) from path position (could be (i, j) or (i, j, t))
                path_pos = pos[:2] if len(pos) > 2 else pos
                # Don't mark goal as visited from path, allow re-entry to goal
                if path_pos != goal:
                    visited.add(path_pos)

        # Expand layer by layer
        for t in range(start_time + 1, end_time):
            prev_layer = reachable[t - 1]
            curr_layer = set()

            for (i, j) in prev_layer:
                for (ni, nj) in adjacency.get((i, j), []):
                    if (ni, nj) not in obstacles and (ni, nj) not in visited:
                        curr_layer.add((ni, nj))
                        visited.add((ni, nj))  # mark as seen globally

            # If no new cells are reachable, you can stop early
            if not curr_layer:
                break

            # To make sure goal is always reachable (and not conflict with goal lock)
            if goal in visited:
                curr_layer.add(goal)

            reachable[t] = curr_layer

        return reachable

    def get_fixed_variables(self):
        M, N = self.problem.grid.M, self.problem.grid.N
        fixed = {}
        robot_nums = self.problem.get_robot_nums()

        # Fix start position at time 0
        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]

            s_i, s_j = robot.current_position
            e_i, e_j = robot.goal

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            end_time = robot.T + start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max
            start_idx = s_i * N + s_j + M * N * start + robot_offset

            fixed[start_idx] = 1
            self.logger.debug(start_idx, "fixed to 1 for robot", robot_id)
            # fix all other time=0 cells
            for i in range(M):
                for j in range(N):
                    n = i * N + j + M * N * start + robot_offset
                    if n != start_idx:
                        fixed[n] = 0

        # Fix goal position at last time step if within window
        # if (self.T + (self.iter * self.t_max)) == self.total_t:
        #     goal_idx = e_i * N + e_j + M * N * (self.T - 1)
        #     fixed[goal_idx] = 1

        # Fix obstacle cells to 0 at all time steps
            for t in range(start, end):
                for (obs_i, obs_j) in self.problem.grid.obstacles:
                    obs_idx = obs_i * N + obs_j + M * N * t + robot_offset
                    fixed[obs_idx] = 0

        # Now we can also fix unreachable cells to 0
        # Based on bfs (this essentially complies with adjacency and tp constraints)
            reachable = self.reachable_positions_aggressive(robot,robot.current_position, start, end)

            # Check if goal is reachable at timestep 1 (one movement away)
            # If so, fix the entire instantaneous path
            if (e_i, e_j) in reachable.get(start + 1, set()):
                self.logger.debug(f"Goal is reachable at timestep 1 for robot {robot_id}. Fixing instantaneous path.")

                # Fix timestep start + 1: goal = 1, all others = 0
                for i in range(M):
                    for j in range(N):
                        n = i * N + j + M * N * (start + 1) + robot_offset
                        if (i, j) == (e_i, e_j):
                            fixed[n] = 1
                            self.logger.debug(f"  Fixed goal position {n} at ({i}, {j}) to 1 at timestep {start + 1}")
                        else:
                            fixed[n] = 0

                # Fix all subsequent timesteps: robot stays at goal
                for t in range(start + 2, end):
                    for i in range(M):
                        for j in range(N):
                            n = i * N + j + M * N * t + robot_offset
                            if (i, j) == (e_i, e_j):
                                fixed[n] = 1
                            else:
                                fixed[n] = 0

            # self.logger.debug(f"Reachable positions for robot {robot_id}: {reachable}")
            # for t in reachable:
            for t in range(start + 1, end):
                for i in range(M):
                    for j in range(N):
                        # By default is there if it visited all nodes it stops the BFS
                        # Instead of returning empty set, return goal (to keep goal lock constraint)
                        if not (i, j) in reachable.get(t, set([(e_i, e_j)])):
                            n = i * N + j + M * N * t + robot_offset
                            fixed[n] = 0

        return fixed

# Backward-compatible alias
QUBOBuilder = GridQUBOBuilder
