from collections import deque

import numpy as np
from .base_qubo import BaseQUBO

GOAL_DISTANCES = ("manhattan", "bfs")
# Fraction of the competing collision penalty a robot's whole-window progress
# reward may reach (see GridQUBOBuilder.progress_weight_cap)
PROGRESS_CAP_MARGIN = 0.5


def compute_obstacle_potential_field(M, N, obstacles, sigma=1.5):
    """
    Compute a 2D potential field where each obstacle contributes
    a Gaussian bump. Sum over all obstacles.
    """
    P = np.zeros((M, N))
    for oi, oj in obstacles:
        # Create grid of distances
        ii, jj = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
        dist_sq = (ii - oi) ** 2 + (jj - oj) ** 2
        P += np.exp(-dist_sq / (2 * sigma**2))
    return P


class GridQUBOBuilder(BaseQUBO):
    def __init__(
        self,
        problem,
        penalties,
        name="grid",
        var_limit=156,  # 101 605 1001
        window_max_steps=None,
        distance_scaling="enhanced_linear",
        robot_window_limits=None,
        verbose_level=2,
        log_reductions=True,
        goal_distance="manhattan",
        obstacle_repulsion=0.4,
        progress_weight=0.0,
        allow_wait=False,
        approach_weight=0.0,
        approach_radius=None,
    ):
        """
        goal_distance: distance the goal-approach reward is built on.
            "manhattan" (default, the original term) ignores obstacles;
            "bfs" is each robot's true shortest-path distance to its goal
            over the static map and its own clearance keep-out (see
            goal_distance_map). Identical on obstacle-free maps.
        obstacle_repulsion: weight of the P_obs potential-field push away
            from obstacles in the same term. It exists to patch Manhattan's
            blindness to walls; with "bfs" it may be redundant (0 disables).
        progress_weight: alpha of a reward linear in progress made this
            window, alpha * (d0 - d(cell)), with d0 the robot's distance at
            the window start. Unlike the distance_scaling curve, which is
            nearly flat far from the goal, it is worth the same per step at
            any distance, and it is bounded by the window length. 0 disables.
            Always capped so the reward a robot can collect over a window
            stays below the collision penalty it competes with (see
            progress_weight_cap); "auto" uses the cap itself.
        allow_wait: let a robot stay in place when it may need to yield, i.e.
            another active robot is within approach_radius at the window
            start (see _may_wait). Staying is then rewarded like a move in the
            adjacency term, and backtracking pairs inside the window are
            dropped -- a pairwise term can't tell "stayed" from "left and came
            back", and the progress term already gives a bounce no net gain.
            Revisits of cells from earlier windows keep K_bt. Off, a wait
            costs roughly K_adj + K_bt per step, far more than a swap's
            2 * K_trail, so windowed multi-robot plans swap instead of yield.
            A robot with nobody near keeps the strict terms: allowing it to
            wait only flattens its landscape.
        approach_weight: soft penalty on two robots ending the window heading
            into each other (see apply_approach_penalty) -- lets a short
            window coordinate route choices it can't see the end of. 0
            disables. approach_radius: path distance beyond which it is 0;
            None (default) = min(M, N) // 2, so on a small map it doesn't
            reach every robot pair and penalise passes in open space.
        """
        if goal_distance not in GOAL_DISTANCES:
            raise ValueError(
                f"goal_distance must be one of {GOAL_DISTANCES}, got {goal_distance!r}"
            )
        self.goal_distance = goal_distance
        self.obstacle_repulsion = obstacle_repulsion
        if progress_weight != "auto" and float(progress_weight) < 0:
            raise ValueError(
                f"progress_weight must be >= 0 or 'auto', got {progress_weight!r}"
            )
        self.progress_weight = progress_weight
        self.allow_wait = allow_wait
        self.approach_weight = approach_weight
        self.approach_radius = (
            approach_radius
            if approach_radius is not None
            else max(1, min(problem.grid.M, problem.grid.N) // 2)
        )
        self._cell_distances = {}
        self._goal_distance_maps = {}
        super().__init__(
            problem,
            penalties,
            name=name,
            var_limit=var_limit,
            window_max_steps=window_max_steps,
            distance_scaling=distance_scaling,
            robot_window_limits=robot_window_limits,
            verbose_level=verbose_level,
            log_reductions=log_reductions,
        )
        self.initial_num_vars = (
            problem.grid.M * problem.grid.N * problem.num_robots * problem.T
        )
        self.P_obs = compute_obstacle_potential_field(
            self.problem.grid.M,
            self.problem.grid.N,
            self.problem.grid.obstacles,
        )
        M, N = problem.grid.M, problem.grid.N
        self._all_grid_cells = [(i, j) for i in range(M) for j in range(N)]
        self.logger.standard("Window max steps:", self.max_window_size())

    def _cells(self, robot_id, t):
        """Return reachable cells for robot at window-relative timestep t.

        Falls back to the full grid when _active_cells has not been populated.
        """
        if self._active_cells is not None:
            return self._active_cells.get((robot_id, t), [])
        return self._all_grid_cells

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
            K_dis = K_goal_approx * (1 / (0.7 + dist_to_goal)) * time_factor

        elif self.distance_scaling == "exponential":
            # Exponential scaling for medium grids
            dist_to_goal = raw_dist * 1.2
            K_dis = K_goal_approx * (1 / (1 + dist_to_goal)) * time_factor
        elif self.distance_scaling == "quadratic":
            dist_to_goal = raw_dist**1.3
            K_dis = K_goal_approx * (1 / (1 + dist_to_goal)) * time_factor

        elif self.distance_scaling == "logarithmic":
            # Logarithmic scaling for balanced approach
            dist_to_goal = np.log(1 + raw_dist * 2)
            K_dis = K_goal_approx * (1 / (1 + dist_to_goal)) * time_factor

        elif self.distance_scaling == "adaptive":
            # Grid-size adaptive scaling
            grid_size = max(self.problem.grid.M, self.problem.grid.N)
            if grid_size <= 3:
                dist_to_goal = raw_dist * 0.4
                K_dis = K_goal_approx * (1 / (0.2 + dist_to_goal)) * time_factor
            elif grid_size <= 5:
                dist_to_goal = raw_dist * 0.8
                K_dis = K_goal_approx * (1 / (0.4 + dist_to_goal)) * time_factor
            else:
                dist_to_goal = raw_dist * 1.2
                K_dis = K_goal_approx * (1 / (0.8 + dist_to_goal)) * time_factor

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
        K_hot = self.penalties["K_hot"]
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
                indices = [
                    i * N + j + (M * N) * t + robot_offset
                    for i, j in self._cells(robot_id, t)
                ]

                for n in indices:
                    self.Q[(n, n)] = self.Q.get((n, n), 0) - K_hot
                for i, n in enumerate(indices):
                    for m in indices[i + 1 :]:
                        self.Q[(n, m)] = self.Q.get((n, m), 0) + 2 * K_hot

    # Be mind the two approaches need different constants to work well
    def apply_adjacency_reward(self):
        """
        Apply adjacency reward: encourage moving to adjacent cells.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        adjacency = self.problem.grid.adjacency
        K_adj = self.penalties["K_adj"]
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
                next_active = set(self._cells(robot_id, t + 1))
                for i, j in self._cells(robot_id, t):
                    # This a consistent linear indexing for the grid
                    n = i * N + j + M * N * t + robot_offset
                    self.Q[(n, n)] = self.Q.get((n, n), 0) + K_adj

                    targets = adjacency[(i, j)]
                    if self._may_wait(robot_id):
                        targets = [*targets, (i, j)]
                    for k, l in targets:
                        if self._active_cells is not None and (k, l) not in next_active:
                            continue
                        m = k * N + l + M * N * (t + 1) + robot_offset
                        self.Q[(n, m)] = self.Q.get((n, m), 0) - K_adj

    def apply_adjacency_penalty(self):
        """
        Apply adjacency penalty: discourage moving to non-adjacent cells.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        adjacency = self.problem.grid.adjacency
        K_adj = self.penalties["K_adj"]
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
                for i, j in self._cells(robot_id, t):
                    n = i * N + j + M * N * t + robot_offset

                    # Look at all possible positions at next time step
                    for k, l in self._cells(robot_id, t + 1):
                        m = k * N + l + M * N * (t + 1) + robot_offset

                        # Skip self-loop unless explicitly allowed
                        if k == i and l == j:
                            continue

                        # Only penalize if (k,l) is NOT in adjacency[(i,j)]
                        if (k, l) not in adjacency[(i, j)]:
                            self.Q[(n, m)] = self.Q.get((n, m), 0) + K_adj

    def apply_start_penalty(self):
        """
        Apply start position penalty: must start at the given position.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_start = self.penalties["K_start"]
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

    def _goal_time_factor(self, t, start):
        """Weight of window step t in the goal-approach term: grows to ~2.5x
        at the window's last step, so the end state counts most."""
        return 1.2 ** (5 * (t - start) / (self.t_max - start))

    def progress_weight_cap(self, start=0):
        """Largest progress weight for which yielding can never cost more
        than PROGRESS_CAP_MARGIN times the collision penalty it avoids (K_trail
        if set, else K_crash). The rule: a soft incentive summed over the
        window must never outweigh the penalty for the violation it could buy
        -- otherwise the QUBO prefers colliding to yielding.

        Yielding one step (waiting instead of colliding) leaves the robot one
        cell behind at every later window step, so it forfeits
        alpha * sum_t time_factor(t) -- not the whole window's progress,
        which grows with the square of the window length and would shrink
        alpha towards 0 on long windows (0.04 at t_max=7, where a lone robot
        on a forced detour just bounced in place)."""
        limit = self.penalties.get("K_trail") or self.penalties.get("K_crash") or 0
        steps = sum(
            self._goal_time_factor(t, start) for t in range(start + 1, self.t_max)
        )
        if not limit or not steps:
            return 0.0
        return PROGRESS_CAP_MARGIN * limit / steps

    def effective_progress_weight(self, start=0):
        """progress_weight clipped to progress_weight_cap ("auto" = the cap)."""
        if not self.progress_weight:
            return 0.0
        cap = self.progress_weight_cap(start)
        if self.progress_weight == "auto":
            return cap
        return min(float(self.progress_weight), cap)

    def _goal_dist(self, robot_id, cell):
        """Distance from `cell` to the robot's goal under goal_distance
        (inf for a cell BFS can't route from)."""
        cell = tuple(cell)
        if self.goal_distance == "bfs":
            return self.goal_distance_map(robot_id).get(cell, float("inf"))
        return self.problem.manhattan_distance(cell, self.problem.robots[robot_id].goal)

    def _may_wait(self, robot_id):
        """allow_wait, but only for a robot with another active robot within
        approach_radius of it at the window start: waiting exists for
        yielding, and a robot with nobody to yield to only gets a flatter
        landscape from it (a lone robot on a forced detour bounced in place)."""
        if not self.allow_wait:
            return False
        near = self.cell_distance_map(self.problem.robots[robot_id].current_position)
        return any(
            tuple(self.problem.robots[other].current_position) in near
            for other in self.get_active_robot_in_window()
            if other != robot_id
        )

    def cell_distance_map(self, cell):
        """{cell: steps} BFS distances from `cell` over grid.adjacency
        (obstacles only). Cached per source cell."""
        cell = tuple(cell)
        if cell not in self._cell_distances:
            adjacency = self.problem.grid.adjacency
            dist = {cell: 0}
            frontier = deque([cell])
            while frontier:
                cur = frontier.popleft()
                if dist[cur] >= self.approach_radius:
                    continue
                for nxt in adjacency.get(cur, ()):
                    if nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        frontier.append(nxt)
            self._cell_distances[cell] = dist
        return self._cell_distances[cell]

    def apply_approach_penalty(self):
        """
        Soft penalty on robot pairs that end the window heading into each
        other: at the window's last step, robot a at c_a and robot b at c_b
        where each lies on a shortest path of the other to its goal,

            d_a(c_a) == dist(c_a, c_b) + d_a(c_b)
            d_b(c_b) == dist(c_a, c_b) + d_b(c_a)

        with d_x the robot's BFS distance-to-goal map. Two such robots must
        pass each other to keep their shortest routes -- a head-on in a
        corridor. A window only sees a few steps, so without this term each
        robot picks between equally short routes independently; on a ring of
        four robots swapping corners only 2 of the 16 choices avoid a
        head-on. Weight fades linearly with dist and is 0 beyond
        approach_radius. Quadratic: it prices positions, not moves (a
        move-direction pair would need four variables).
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        robot_nums = self.problem.get_robot_nums()
        t_end = self.t_max - 1
        active = self.get_active_robots_per_timestep_in_window().get(
            self.current_T + t_end, []
        )
        radius = self.approach_radius
        for a_idx, a in enumerate(active):
            d_a = self.goal_distance_map(a)
            off_a = robot_nums[a] * (M * N * self.total_t)
            for b in active[a_idx + 1 :]:
                d_b = self.goal_distance_map(b)
                off_b = robot_nums[b] * (M * N * self.total_t)
                cells_b = list(self._cells(b, t_end))
                for c_a in self._cells(a, t_end):
                    near = self.cell_distance_map(c_a)
                    for c_b in cells_b:
                        dab = near.get(c_b)
                        if not dab or c_a not in d_a or c_b not in d_b:
                            continue  # same cell (crash term) or out of range
                        if d_a[c_a] != dab + d_a.get(c_b, -1):
                            continue
                        if d_b[c_b] != dab + d_b.get(c_a, -1):
                            continue
                        w = self.approach_weight * (1 - dab / (radius + 1))
                        idx_a = c_a[0] * N + c_a[1] + M * N * t_end + off_a
                        idx_b = c_b[0] * N + c_b[1] + M * N * t_end + off_b
                        key = (min(idx_a, idx_b), max(idx_a, idx_b))
                        self.Q[key] = self.Q.get(key, 0.0) + w

    def goal_distance_map(self, robot_id):
        """{(i, j): steps} shortest-path distance from every cell to this
        robot's goal, by BFS backwards from the goal over grid.adjacency
        (obstacle-aware), skipping the robot's clearance keep-out cells.
        Cells that can't reach the goal are absent, so they get no reward.

        Unlike Manhattan distance it has no local minima: every reachable
        non-goal cell has a neighbour one step closer. That makes it the exact
        cost-to-go for a robot on its own, so a short window scored with it
        can't be lured into a dead end or the wrong side of a wall. It knows
        nothing about other robots -- the QUBO still decides all of that.
        Computed once per robot and cached: map and goal don't change.
        """
        if robot_id not in self._goal_distance_maps:
            goal = tuple(self.problem.robots[robot_id].goal)
            keepout = self.problem.get_obstacle_keepout().get(robot_id, frozenset())
            adjacency = self.problem.grid.adjacency
            dist = {goal: 0}
            frontier = deque([goal])
            while frontier:
                cell = frontier.popleft()
                for nxt in adjacency.get(cell, ()):
                    if nxt not in dist and nxt not in keepout:
                        dist[nxt] = dist[cell] + 1
                        frontier.append(nxt)
            self._goal_distance_maps[robot_id] = dist
        return self._goal_distance_maps[robot_id]

    def apply_goal_approximation_penalty(self, robot_id):
        """
        Apply goal approximation penalty: encourage getting near the goal.
        This is used when reaching goal is not possible in a single window.
        Uses a more balanced approach between goal attraction and obstacle
        avoidance.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_goal_approx = self.penalties["K_goal_approx"]
        K_obs_repel = self.obstacle_repulsion
        robot_nums = self.problem.get_robot_nums()

        robot_offset = robot_nums[robot_id] * (M * N * self.total_t)

        robot = self.problem.robots[robot_id]

        start_time = robot.start_time
        start = 0
        if self.current_T < start_time:
            start = start_time - self.current_T

        d0 = self._goal_dist(robot_id, robot.current_position)
        alpha = self.effective_progress_weight(start)

        for t in range(start + 1, self.t_max):
            time_factor = self._goal_time_factor(t, start)
            for i, j in self._cells(robot_id, t):
                n = i * N + j + M * N * t + robot_offset

                # Skip obstacle cells entirely - let explicit obstacle
                # constraint handle them
                if (i, j) in self.problem.grid.obstacles:
                    continue

                # Goal progress (time-weighted), shaped by distance_scaling
                raw_dist = self._goal_dist(robot_id, (i, j))
                K_dis = self.calculate_manhattan_penalty(
                    raw_dist, K_goal_approx, time_factor
                )
                if alpha and max(d0, raw_dist) != float("inf"):
                    K_dis += alpha * time_factor * (d0 - raw_dist)

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
        K_goal = self.penalties["K_goal"]

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
        K_goal = self.penalties["K_goal"]
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
                    if (goal_idx, goal_idx) not in self.Q:
                        continue  # goal not reachable at this timestep — nothing to bias
                    # Note that since I initially considered all this time factor for single robot starting in t=-
                    # I adjust to keep the same growth by reducing time start to both)
                    time_factor = 1 + ((t - start) / (end - start))
                    self.Q[(goal_idx, goal_idx)] += (
                        -K_goal * time_factor * window_constant
                    )

    def apply_goal_early_penalty(self):
        """
        Apply early goal penalty: encourage reaching the goal earlier.
        It's stronger early and then decreases over time (till it reaches normal goal penalty).
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        e_i, e_j = self.problem.end
        K_goal = self.penalties["K_goal"]
        for t in range(1, self.T):
            goal_idx = e_i * N + e_j + M * N * t
            time_factor = 1 + (self.T - t) / self.T
            self.Q[(goal_idx, goal_idx)] += -K_goal * time_factor

    def apply_lock_after_goal(self):
        """
        Apply lock after goal: discourage leaving the goal position once reached.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_lock = self.penalties["K_lock"]
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
                # Skip if goal is not a reachable cell at this timestep — avoids phantom Q entries
                if (e_i, e_j) not in self._cells(robot_id, t):
                    continue
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
        K_bt = self.penalties["K_bt"]
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

            # Per-timestep active sets for O(1) membership checks
            active_sets = {t: set(self._cells(robot_id, t)) for t in range(start, end)}

            all_reachable = set()
            for t in range(start, end):
                all_reachable.update(active_sets[t])

            for i, j in all_reachable:
                # Skip goal — allow multiple visits
                if i == e_i and j == e_j:
                    continue

                # With aggressive BFS each cell appears at exactly one timestep, so
                # iterating range(start, end) blindly would create phantom Q entries for
                # timesteps where the cell is not active, leaking variables into the QUBO.
                active_ts = [t for t in range(start, end) if (i, j) in active_sets[t]]
                if self._may_wait(robot_id):
                    active_ts = []  # see allow_wait in __init__
                for idx1, t1 in enumerate(active_ts):
                    g_t = i * N + j + M * N * t1 + robot_offset
                    for t2 in active_ts[idx1 + 1 :]:
                        g_t2 = i * N + j + M * N * t2 + robot_offset
                        self.Q[(g_t, g_t2)] = self.Q.get((g_t, g_t2), 0) + K_bt

            if robot.active and robot.path:
                len_sol = len(robot.path)
                for t in range(start, end):
                    for p_idx, pos in enumerate(robot.path):
                        i, j = pos[:2]
                        if (i, j) not in active_sets[t]:
                            continue
                        idx = i * N + j + M * N * t + robot_offset
                        time_factor = (1 + (len_sol - p_idx)) / len_sol
                        self.Q[(idx, idx)] = (
                            self.Q.get((idx, idx), 0) + K_bt * time_factor
                        )

    def apply_tp_penalty(self):
        """
        Apply a penalty if goal is reached before is even physically possible
        (i.e. if the goal is reached at time step t, but the manhattan distance is T)
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_tp = self.penalties["K_tp"]
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
                if (goal_idx, goal_idx) not in self.Q:
                    continue  # goal not reachable yet — BFS already enforces this
                self.Q[(goal_idx, goal_idx)] += K_tp  # Penalty for arriving to soon

    def apply_terrain_penalty(self):
        """
        Apply terrain penalty: encourage moving to cells with lower terrain cost.
        It introduces a linear bias depending on material costs in the grid.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_ter = self.penalties["K_ter"]
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
                for i, j in self._cells(robot_id, t):
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
        K_elev = self.penalties["K_elev"]
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
                next_active = set(self._cells(robot_id, t + 1))
                for i, j in self._cells(robot_id, t):
                    hi = self.problem.grid.get_elevation_at(i, j)
                    n = (
                        i * N + j + M * N * t + robot_offset
                    )  # linear index for time step t

                    for k, l in adjacency[(i, j)]:
                        if self._active_cells is not None and (k, l) not in next_active:
                            continue
                        hk = self.problem.grid.get_elevation_at(k, l)
                        delta_h = hk - hi  # positive = uphill

                        m = (
                            k * N + l + M * N * (t + 1) + robot_offset
                        )  # next time step index

                        if delta_h > 0:
                            move_cost = K_elev * (
                                delta_h**1.8
                            )  # super-linear for steep climbs
                        elif delta_h < -0.7:
                            move_cost = (
                                K_elev * 0.7 * abs(delta_h)
                            )  # still costly if too steep down
                        else:
                            move_cost = (
                                K_elev * 0.3 * abs(delta_h)
                            )  # mild descent = easy

                        # Add to QUBO: only if move occurs
                        self.Q[(n, m)] = self.Q.get((n, m), 0) + move_cost

    def apply_obstacle_penalty(self):
        """
        Apply hard obstacle penalty: strongly discourage or prohibit entering
        obstacle cells. This creates a much stronger constraint than the soft
        potential field approach.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_obs = self.penalties.get("K_obs", 2)
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

            # BFS-restricted windows already exclude obstacle cells from active_cells
            # (reachable_positions_aggressive filters them out), so this hard penalty
            # is redundant there and would only leak unconstrained phantom variables
            # into Q. Only needed when building without active-cell restriction.
            # It is caused even with K_obs = 0, that happens because of the indexing .get that always create a var (and occupies a Qubit)
            if self._active_cells is not None:
                continue

            for t in range(start, end):
                for obs_i, obs_j in self.problem.grid.obstacles:
                    obs_idx = obs_i * N + obs_j + M * N * t + robot_offset
                    self.Q[(obs_idx, obs_idx)] = (
                        self.Q.get((obs_idx, obs_idx), 0) + K_obs
                    )

    def _clearance_shell(self, robot_id1, robot_id2):
        """D_ab for this pair (quantum/utils/clearance.py), or {(0,0)} when
        clearance isn't configured.
        The default case for synthetic maps reduces every shell loop below to
        plain same-cell matching (this is a no-op at defaults)."""
        return self.problem.get_clearance_table().get(
            (robot_id1, robot_id2)
        ) or frozenset({(0, 0)})

    def apply_crash_penalty(self):
        M, N = self.problem.grid.M, self.problem.grid.N
        K_crash = self.penalties.get("K_crash", 0)
        robot_nums = self.problem.get_robot_nums()
        active_robots_per_timestep = self.get_active_robots_per_timestep_in_window()
        for t, active_robots in active_robots_per_timestep.items():
            if len(active_robots) < 2:
                continue  # no collision possible
            t_window = t - self.current_T
            # Note that if I don't substract current_t it doesn't keep relative window time
            for robot_id1 in active_robots:
                for robot_id2 in active_robots:
                    if robot_nums[robot_id1] >= robot_nums[robot_id2]:
                        continue

                    robot_offset1 = robot_nums[robot_id1] * (M * N * self.total_t)
                    robot_offset2 = robot_nums[robot_id2] * (M * N * self.total_t)

                    # Only cells robot1 can reach whose D_ab shell robot2 can
                    # also reach — the only place a footprint overlap is
                    # possible. D_ab = {(0,0)} (the default) collapses this to
                    # plain same-cell matching, i.e. the pre-clearance model.
                    D = self._clearance_shell(robot_id1, robot_id2)
                    cells2 = set(self._cells(robot_id2, t_window))
                    for i, j in self._cells(robot_id1, t_window):
                        idx1 = i * N + j + M * N * t_window + robot_offset1
                        for di, dj in D:
                            w = (i - di, j - dj)
                            if w in cells2:
                                idx2 = (
                                    w[0] * N + w[1] + M * N * t_window + robot_offset2
                                )
                                self.Q[(idx1, idx2)] = (
                                    self.Q.get((idx1, idx2), 0) + K_crash
                                )

    def apply_trailing_penalty(self):
        """
        Robot-robot separation penalty: same-time footprint overlap (the
        crash case, D_ab-shell generalised) plus the cross-time trailing
        hazard (K_trail).
        Note that trailing also guards for swap and is safer, meaning that
        is reduces more search space which is not necessarily good but
        is easier to model in QUBO and is a safe decision for real robot deployment
        where swap may not be enough.

        P = K_crash * [overlap at t] + K_trail * [overlap between (t,t+1) and (t+1,t)]

        Summed over each robot pair's D_ab shell. The same-time term reuses
        K_crash as-is (identical to apply_crash_penalty), while the cross-time
        terms get their own K_trail.

        This method was named apply_swap_penalty / K_swap before —
        renamed because it never computed swap: a quadratic-only penalty
        cannot express swap's genuine AND-of-both-endpoints condition (that
        needs 4 variables jointly, i.e. ancillas) without leaving quadratic
        form, so these pairwise-product terms always computed trailing's
        OR-of-either-endpoint predicate instead, under the wrong name. QUBO
        has no exact-swap tier at all, unlike CBS/ILP, for exactly this
        reason. K_trail is the knob to tune down if the false-positive cost
        (this also fires when one robot simply follows another into a
        just-vacated footprint, which isn't an actual collision — an
        accepted overconstraint, same as CBS/ILP's trailing) outweighs the
        benefit, independently of crash protection.

        When K_trail is active it replaces apply_crash_penalty (see build())
        so the same-time term isn't double counted.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        K_crash = self.penalties.get("K_crash", 0)
        K_trail = self.penalties.get("K_trail", 0)
        robot_nums = self.problem.get_robot_nums()
        active_robots_per_timestep = self.get_active_robots_per_timestep_in_window()

        for t, active_robots in active_robots_per_timestep.items():
            if len(active_robots) < 2:
                continue
            t_window = t - self.current_T
            next_active_robots = active_robots_per_timestep.get(t + 1, [])

            for robot_id1 in active_robots:
                for robot_id2 in active_robots:
                    if robot_nums[robot_id1] >= robot_nums[robot_id2]:
                        continue

                    robot_offset1 = robot_nums[robot_id1] * (M * N * self.total_t)
                    robot_offset2 = robot_nums[robot_id2] * (M * N * self.total_t)
                    D = self._clearance_shell(robot_id1, robot_id2)

                    # Same-cell-shell, same-time term (native crash constraint)
                    cells1_t = list(self._cells(robot_id1, t_window))
                    cells2_t = set(self._cells(robot_id2, t_window))
                    for i, j in cells1_t:
                        idx1 = i * N + j + M * N * t_window + robot_offset1
                        for di, dj in D:
                            w = (i - di, j - dj)
                            if w in cells2_t:
                                idx2 = (
                                    w[0] * N + w[1] + M * N * t_window + robot_offset2
                                )
                                self.Q[(idx1, idx2)] = (
                                    self.Q.get((idx1, idx2), 0) + K_crash
                                )

                    # Cross-time trailing terms, only meaningful if both robots
                    # are still active in the window at t+1
                    if (
                        robot_id1 not in next_active_robots
                        or robot_id2 not in next_active_robots
                    ):
                        continue
                    t_next_window = t_window + 1
                    cells1_next = set(self._cells(robot_id1, t_next_window))
                    cells2_next = set(self._cells(robot_id2, t_next_window))

                    # robot1 arrives at t+1 inside robot2's D_ab shell at t
                    for i, j in cells1_next:
                        idx1 = i * N + j + M * N * t_next_window + robot_offset1
                        for di, dj in D:
                            w = (i - di, j - dj)
                            if w in cells2_t:
                                idx2 = (
                                    w[0] * N + w[1] + M * N * t_window + robot_offset2
                                )
                                self.Q[(idx1, idx2)] = (
                                    self.Q.get((idx1, idx2), 0) + K_trail
                                )

                    # robot2 arrives at t+1 inside robot1's D_ab shell at t.
                    # Same "w = anchor - d" convention as above, anchored on
                    # robot1's t-cell this time (D_ab is centrally symmetric
                    # for circular robots, so the same D serves both
                    # directions).
                    for i, j in cells1_t:
                        idx1 = i * N + j + M * N * t_window + robot_offset1
                        for di, dj in D:
                            w = (i - di, j - dj)
                            if w in cells2_next:
                                idx2 = (
                                    w[0] * N
                                    + w[1]
                                    + M * N * t_next_window
                                    + robot_offset2
                                )
                                self.Q[(idx1, idx2)] = (
                                    self.Q.get((idx1, idx2), 0) + K_trail
                                )

    def build(self, constraints_to_apply=None):
        self._warn_if_unrestricted_build(len(self._all_grid_cells))
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
                "K_crash": "crash",
                "K_trail": "trailing",
            }
            constraints_to_apply = [
                v for k, v in penalty_to_constraint.items() if k in self.penalties
            ]
            # apply_trailing_penalty already applies the same-cell/same-time
            # term itself (weighted by K_crash), so don't also run
            # apply_crash_penalty separately — that would double-count it.
            if "trailing" in constraints_to_apply and "crash" in constraints_to_apply:
                constraints_to_apply.remove("crash")

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
        if "crash" in constraints_to_apply:
            self.apply_crash_penalty()
        if "trailing" in constraints_to_apply:
            self.apply_trailing_penalty()
        if self.approach_weight:
            self.apply_approach_penalty()

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
        adjacency = (
            self.problem.grid.adjacency
        )  # Note that my adjacency map don't include obstacles
        obstacles = self.problem.grid.obstacles

        if obstacles is None:
            obstacles = []

        obstacles = set(obstacles)
        reachable = {start_time: {start}}

        for t in range(start_time + 1, end_time):
            prev_layer = reachable[t - 1]
            curr_layer = set()

            for i, j in prev_layer:
                for ni, nj in adjacency.get((i, j), []):
                    if (ni, nj) not in obstacles:
                        curr_layer.add((ni, nj))

            reachable[t] = curr_layer

        return reachable

    def reachable_positions_aggressive(
        self, robot, start, start_time, end_time, blocked=None, allow_wait_at=None
    ):
        """
        Compute reachable positions per time step without backtracking.
        That means once a cell is reached, it won't be revisited in future time steps.

        Args:
            robot: Robot object with current_position.
            start_time (int): starting time step.
            end_time (int): ending time step (exclusive).
            blocked: optional {t: {(i, j), ...}} of cells to treat as
                temporary/dynamic obstacles at that specific timestep only --
                used when recalculating around a detected collision, so the
                colliding cell can't be re-derived here and handed back as
                "the only option" again. Unlike `obstacles`, this is scoped
                to one timestep, not the whole grid for all time.
            allow_wait_at: optional set of timesteps at which, if blocking
                and "no revisit" leave zero forward candidates, the robot is
                allowed to stay at wherever it already was instead of
                terminating the walk early -- a one-step, targeted opt-in to
                reachable_positions_safe()'s semantics, used as the fallback
                when dynamic-obstacle rerouting alone finds no escape. Normal
                aggressive (no-revisit) expansion resumes at the next step.

        Returns:
            dict[int, set[tuple[int, int]]]: {t: {(i, j), ...}} reachable positions per time step.
        """

        goal = robot.goal
        adjacency = self.problem.grid.adjacency  # adjacency map without obstacles
        obstacles = set(self.problem.grid.obstacles or [])
        blocked = blocked or {}
        allow_wait_at = allow_wait_at or set()

        reachable = {start_time: {start}}
        visited = {start}  # <- prevent backtracking / revisiting

        # Expand layer by layer
        for t in range(start_time + 1, end_time):
            prev_layer = reachable[t - 1]
            curr_layer = set()
            blocked_at_t = blocked.get(t, set())

            for i, j in prev_layer:
                for ni, nj in adjacency.get((i, j), []):
                    if (
                        (ni, nj) not in obstacles
                        and (ni, nj) not in visited
                        and (ni, nj) not in blocked_at_t
                    ):
                        curr_layer.add((ni, nj))
                        visited.add((ni, nj))  # mark as seen globally

            # Always include goal once reachable — robot may stay there indefinitely.
            # This must come before the empty check so late timesteps get {goal} instead of {}.
            if goal in visited and goal not in blocked_at_t:
                curr_layer.add(goal)

            if not curr_layer:
                # Nowhere new to go this step -- stay put rather than ending
                # the walk here, then resume normal aggressive expansion
                # from the same cells next step. Still respects blocked_at_t:
                # if the "stay" cell is itself a confirmed collision (another
                # robot is fixed there too), offering it again would just
                # get rejected identically on every retry -- exclude it like
                # any other blocked cell instead of looping on it.
                stay_layer = (
                    set(prev_layer) - blocked_at_t if t in allow_wait_at else set()
                )
                if stay_layer:
                    curr_layer = stay_layer
                else:
                    # Stop only if goal hasn't been reached yet and no new
                    # cells found (and waiting wasn't offered/didn't help).
                    break

            reachable[t] = curr_layer
        self.logger.debug(
            f"Reachable positions for robot {robot.robot_id}: {reachable}"
        )
        return reachable

    def reachable_positions_safe(self, robot, start, start_time, end_time):
        """
        Monotone reachability -- staying in place and revisiting are allowed.

        Same semantics as ILPBuilder.bfs_reachable_sets(): the set at t is the
        set at t-1 unioned with its neighbours, so it never excludes a cell the
        robot could legitimately occupy. reachable_positions_aggressive()
        forces a brand-new, never-revisited cell at every timestep, which
        forbids waiting -- and waiting is how robots yield to each other, so
        that pruning can make a solvable multi-robot instance unsolvable.

        The set saturates once it covers the robot's connected component, so
        this stops growing early rather than recomputing an unchanged set.
        """
        adjacency = self.problem.grid.adjacency
        obstacles = set(self.problem.grid.obstacles or [])

        current = {start}
        reachable = {start_time: set(current)}
        for t in range(start_time + 1, end_time):
            grown = current | {
                n
                for cell in current
                for n in adjacency.get(cell, [])
                if n not in obstacles
            }
            if len(grown) == len(current):
                # saturated: every later timestep has the same set
                for rest in range(t, end_time):
                    reachable[rest] = set(current)
                break
            current = grown
            reachable[t] = set(current)
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

            for i, j in prev_layer:
                for ni, nj in adjacency.get((i, j), []):
                    if (ni, nj) not in obstacles and (ni, nj) not in visited:
                        curr_layer.add((ni, nj))
                        visited.add((ni, nj))  # mark as seen globally

            # Always include goal once reachable — robot may stay there indefinitely.
            # Must come before the empty check so late timesteps get {goal} instead of {}.
            if goal in visited:
                curr_layer.add(goal)

            # Stop only if goal hasn't been reached yet and no new cells found
            if not curr_layer:
                break

            reachable[t] = curr_layer

        return reachable

    def get_logical_variables(self, bfs_variant=None):
        """
        Returns (fixed_ones, active_cells):
        - fixed_ones: {flat_idx: 1} for variables known to be 1 (starts, instantaneous paths).
          Absent entries are implicitly 0 in the global binary encoding — no zeros stored.
        - active_cells: {(robot_id, t): [(i, j), ...]} of cells that may be 1, built
          directly from BFS reachability without an inversion scan over the full grid.
        """
        M, N = self.problem.grid.M, self.problem.grid.N
        fixed_ones = {}
        active_cells = {}
        robot_nums = self.problem.get_robot_nums()

        for robot_id in self.get_active_robot_in_window():
            robot_offset = robot_nums[robot_id] * (M * N * self.total_t)
            robot = self.problem.robots[robot_id]

            s_i, s_j = robot.current_position
            e_i, e_j = robot.goal

            start_time = robot.start_time
            start = 0
            if self.current_T < start_time:
                start = start_time - self.current_T
            end_time = robot.T + robot.start_time
            end = end_time - self.current_T
            if end_time > self.current_T + self.t_max:
                end = self.t_max

            start_idx = s_i * N + s_j + M * N * start + robot_offset
            fixed_ones[start_idx] = 1
            self.logger.debug(start_idx, "fixed to 1 for robot", robot_id)

            reachable = self.reachable_for_window(
                robot, robot.current_position, start, end, bfs_variant
            )

            if (e_i, e_j) in reachable.get(start + 1, set()):
                self.logger.standard(
                    f"Goal is reachable at timestep 1 for robot {robot_id}. Fixing instantaneous path."
                )
                active_cells[(robot_id, start)] = [(s_i, s_j)]
                for t in range(start + 1, end):
                    goal_idx = e_i * N + e_j + M * N * t + robot_offset
                    fixed_ones[goal_idx] = 1
                    if t == start + 1:
                        self.logger.debug(
                            f"  Fixed goal position {goal_idx} at ({e_i}, {e_j}) to 1 at timestep {t}"
                        )
                    active_cells[(robot_id, t)] = [(e_i, e_j)]
            else:
                active_cells[(robot_id, start)] = [(s_i, s_j)]
                for t in range(start + 1, end):
                    # sorted(), not list(): reachable.get(t, ...) is a set, and
                    # its iteration order depends on PYTHONHASHSEED. That order
                    # flows into Q insertion order and the diagonal fixer's
                    # candidate order, so an unsorted list here makes a
                    # corridor-yield tie between two robots resolve differently
                    # per process — the whole windowed solve then lands on a
                    # valid or invalid path purely by hash seed.
                    active_cells[(robot_id, t)] = sorted(reachable.get(t, {(e_i, e_j)}))

        return fixed_ones, active_cells


# Backward-compatible alias
QUBOBuilder = GridQUBOBuilder
