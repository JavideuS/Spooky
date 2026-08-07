from typing import Any, Dict
from .base_solver import BaseSolver
from .cbs_algorithm import ConflictBasedSearch


class CBSSolver(BaseSolver):
    """
    Classical MAPF baseline via Conflict-Based Search (CBS): space-time A*
    per robot (cbs_algorithm.SpaceTimeAStar) as the low-level solver, with a
    high-level constraint-tree search resolving conflicts by branching. Works
    with either GridCBSBuilder or GraphCBSBuilder. This class never branches on grid vs. graph itself,
    it just asks the builder for vars_per_time, local_index(v), and
    get_robot_start_goal(robot_id) to translate between CBS's plain-graph-node-id paths and the same flat
    (robot, time, position) index QUBO/ILP solvers use, so decode_path() and
    every other BaseSolver post-processing helper work unchanged.
    """

    def __init__(
        self,
        node_limit=5000,
        time_limit=60,
        normalize_scale=0,
        num_reads=1,
        max_corrections=0,
        verbose_level=2,
        **kwargs,
    ):
        """
        Args:
            node_limit: Max constraint-tree nodes to expand (default 5000).
                Safety cap so a pathological conflict-branching case can't
                hang a sweep — mirrors ILPSolver's time_limit in spirit.
            time_limit: Max wall-clock seconds (default 60). Whichever of
                node_limit/time_limit is hit first stops the search — but
                unlike ILPSolver's HiGHS time_limit, this does NOT return a
                best-incumbent-so-far: CBS has no such concept (see
                ConflictBasedSearch.solve()'s docstring — every node it
                pops either has zero conflicts, and gets returned
                immediately, or at least one, and there's no "good but
                unproven" state in between). A budget cutoff here means an
                empty solution and energy=inf, not a worse-but-valid answer.
        """
        super().__init__(
            solver="cbs",
            normalize_scale=normalize_scale,
            num_reads=num_reads,
            max_corrections=max_corrections,
            verbose_level=verbose_level,
            node_limit=node_limit,  # forwarded so to_dict()/the manifest records these too
            time_limit=time_limit,
            **kwargs,
        )
        self.node_limit = node_limit
        self.time_limit = time_limit

    def solve(self, builder, optimization=False, preprocess=True) -> Dict[str, Any]:
        """
        Solve the MAPF problem held by builder via CBS.

        Args:
            builder: GridCBSBuilder or GraphCBSBuilder instance
            optimization: Accepted for interface compatibility; unused (CBS
                is exact/deterministic search, no variational step).
            preprocess: When True (default), builder.build() computes
                forward+backward BFS reachability per robot and CBS's
                low-level search only considers those cells — see
                BaseCBSBuilder.build(). When False, the search considers
                every free cell at every step. Always rebuilds (even if
                builder.graph already exists) so this flag reliably takes
                effect, same reasoning as ILPSolver.solve().

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        builder.build(preprocess=preprocess)

        problem = builder.problem
        robot_nums = problem.get_robot_nums()
        T = problem.T
        vars_per_time = builder.vars_per_time
        total_vars = vars_per_time * T * problem.num_robots

        robots_meta = {}
        for robot_id, robot in problem.robots.items():
            start_node, goal_node = builder.get_robot_start_goal(robot_id)
            robots_meta[robot_id] = {
                "start": start_node,
                "goal": goal_node,
                "start_time": robot.start_time,
                "deadline": robot.start_time + robot.T - 1,
            }

        cbs = ConflictBasedSearch(
            builder.graph, node_limit=self.node_limit, time_limit=self.time_limit
        )
        paths, meta = cbs.solve(robots_meta, legal_cells=builder.legal_cells)

        solution = {idx: 0 for idx in range(total_vars)}
        for robot_id, path in paths.items():
            robot_num = robot_nums[robot_id]
            robot_offset = robot_num * (vars_per_time * T)
            for node_id, t in path:
                solution[robot_offset + t * vars_per_time + node_id] = 1

        # Sum-of-costs: each robot's steps away from goal (arrival_time -
        # start_time), not padded path length
        if len(paths) < problem.num_robots:
            energy = float("inf")  # some robot has no path at all
        else:
            energy = sum(
                self._robot_cost(
                    path, robots_meta[rid]["goal"], robots_meta[rid]["start_time"]
                )
                for rid, path in paths.items()
            )

        self.logger.standard(
            f"CBS solve complete: {meta['termination_condition']}, "
            f"energy={energy}, nodes_expanded={meta['nodes_expanded']}"
        )

        # Write the solved paths back onto problem.robots — QUBO solvers do
        # this as a side effect of their windowed loop; CBS has no windowing
        # loop, so it has to happen explicitly here, same as ILPSolver.
        decoded = self.decode_path(solution, problem)
        num_to_id = {num: rid for rid, num in robot_nums.items()}
        for robot_num, coords in self.get_robot_paths(decoded).items():
            if not coords:
                continue  # robot has no path (infeasible) — leave its state untouched
            robot = problem.robots[num_to_id[robot_num]]
            robot.path = coords
            robot.current_position = coords[-1][:2]
            robot.active = False

        return {
            "solution": solution,
            "energy": energy,
            "raw_response": meta,
            "metadata": {
                "termination_condition": meta["termination_condition"],
                "solver_config": self.to_dict(),
                # Single-entry list, not a per-window loop — CBS solves the
                # whole horizon in one shot. Wrapped in a list so
                # BenchmarkRunner.run_build()'s existing window_stats
                # aggregation (benchmark.py) picks it up unchanged
                "window_stats": [builder.bfs_stats],
            },
        }

    @staticmethod
    def _robot_cost(path, goal, start_time):
        t_arrive = path[-1][1]
        for node, t in reversed(path):
            if node != goal:
                break
            t_arrive = t
        return t_arrive - start_time
