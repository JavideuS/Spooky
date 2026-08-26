import inspect
import pyomo.environ as pyo
from typing import Any, Dict
from .base_solver import BaseSolver
from quantum.utils import preprocess as preprocess_modes


class ILPSolver(BaseSolver):
    """
    Exact MAPF solver via Pyomo ILP (hard constraints, no penalties). Works
    with either GridILPBuilder or GraphILPBuilder — this class never branches
    on grid vs. graph itself, it just asks the builder for vars_per_time and
    local_index(v) to translate a solved Pyomo variable into a flat index.
    Solves the whole time horizon in a single shot — no windowing, no
    sampling — then packs the result into the same flat (robot, time,
    position) variable index QUBO solvers use, so decode_path() and every
    other BaseSolver post-processing helper work unchanged.
    """

    def __init__(
        self,
        pyomo_solver_name="appsi_highs",
        normalize_scale=0,
        num_reads=1,
        max_corrections=0,
        verbose_level=2,
        time_limit=30,
        **kwargs,
    ):
        """
        Args:
            time_limit: Max solve time in seconds (default 30). Nothing else
                in this codebase bounds ILP's runtime — unlike QAOA's fixed
                opt_steps budget, branch-and-bound can otherwise run
                arbitrarily long as problems scale up. If the limit is hit
                before HiGHS proves optimality, it returns its best
                incumbent so far (termination_condition reflects this —
                check it before trusting `energy` as the true optimum).
        """
        super().__init__(
            solver="ilp",
            normalize_scale=normalize_scale,
            num_reads=num_reads,
            max_corrections=max_corrections,
            verbose_level=verbose_level,
            pyomo_solver_name=pyomo_solver_name,
            time_limit=time_limit,  # forwarded so to_dict()/the manifest records it too
            **kwargs,
        )
        self.pyomo_solver_name = pyomo_solver_name
        self.time_limit = time_limit

    def solve(self, builder, optimization=False, preprocess=True) -> Dict[str, Any]:
        """
        Solve the ILP model held by builder.

        Args:
            builder: GridILPBuilder or GraphILPBuilder instance
            optimization: Accepted for interface compatibility; unused (no
                variational step for an exact ILP solve).
            preprocess: When True (default), builder.build() fixes x[a, v, t]
                to 0 for every BFS-from-start-unreachable (v, t) — see
                BaseILPBuilder.build(). When False, only the per-robot active
                time window is fixed; the full free-cell set stays open.
                Always rebuilds (even if builder.model already exists) so
                this flag reliably takes effect regardless of any prior
                build() call — no windowing/sampling either way, ILP solves
                the whole horizon exactly in one shot.

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        # coerce the mode to the bool this builder expects; "raw" is
        # truthy, so passing the mode through would prune in raw mode
        builder.build(preprocess=preprocess_modes.applies_bfs_pruning(preprocess))

        pyomo_solver = pyo.SolverFactory(self.pyomo_solver_name)
        solve_params = inspect.signature(pyomo_solver.solve).parameters
        solve_kwargs = {}
        if "timelimit" in solve_params:
            # APPSI-backed solvers (appsi_highs, appsi_cbc, ...) — Pyomo's
            # LegacySolverInterface.solve() replaces self.config with a fresh
            # default config on every call and only reads time_limit from
            # this kwarg, so setting pyomo_solver.config.time_limit ahead of
            # time (the obvious-looking approach) is silently discarded.
            solve_kwargs["timelimit"] = self.time_limit
        else:
            # Plain legacy plugin backends (cbc, glpk, ...) — no shared
            # kwarg; option name varies by solver (cbc wants 'seconds',
            # glpk wants 'tmlim'). 'time_limit' covers the common case but
            # isn't universal for every possible backend.
            pyomo_solver.options["time_limit"] = self.time_limit

        # Load the solution ourselves rather than letting solve() do it. When
        # the time limit fires before HiGHS has any incumbent there is nothing
        # to load, and the default load_solutions=True raises a RuntimeError
        # that kills the whole run -- so a timeout, which is an ordinary
        # censored result, would abort a sweep instead of being recorded.
        defers_loading = "load_solutions" in solve_params
        if defers_loading:
            solve_kwargs["load_solutions"] = False

        results = pyomo_solver.solve(builder.model, **solve_kwargs)
        termination = str(results.solver.termination_condition)

        if defers_loading:
            try:
                builder.model.solutions.load_from(results)
            except (RuntimeError, ValueError, AttributeError, IndexError) as exc:
                self.logger.minimal(
                    f"ILP returned no loadable solution ({termination}): {exc}. "
                    "Reporting an empty path rather than raising -- validation "
                    "will mark the run invalid and termination_condition "
                    "records why."
                )
                vars_per_time = builder.vars_per_time
                total = vars_per_time * builder.problem.T * builder.problem.num_robots
                return {
                    "solution": {idx: 0 for idx in range(total)},
                    "energy": float("inf"),
                    "raw_response": results,
                    "metadata": {
                        "termination_condition": termination,
                        "solver_config": self.to_dict(),
                        "window_stats": [],
                    },
                }

        problem = builder.problem
        robot_nums = problem.get_robot_nums()
        T = problem.T
        vars_per_time = builder.vars_per_time
        total_vars = vars_per_time * T * problem.num_robots

        solution = {idx: 0 for idx in range(total_vars)}
        for a in builder.model.A:
            robot_num = robot_nums[a]
            robot_offset = robot_num * (vars_per_time * T)
            for t in builder.model.T:
                for v in builder.model.V:
                    if pyo.value(builder.model.x[a, v, t]) > 0.5:
                        solution[robot_offset + t * vars_per_time + builder.local_index(v)] = 1
                        break

        self.logger.standard(
            f"ILP solve complete: {results.solver.termination_condition}, "
            f"energy={pyo.value(builder.model.obj)}"
        )

        # Write the solved paths back onto problem.robots — QUBO solvers do
        # this as a side effect of their windowed loop (builder.update_problem());
        # ILP has no windowing loop, so it has to happen explicitly here. Callers
        # (BenchmarkRunner, qubo_cli.py's output printing) read robot.path directly.
        path = self.decode_path(solution, problem)
        num_to_id = {num: rid for rid, num in robot_nums.items()}
        for robot_num, coords in self.get_robot_paths(path).items():
            robot = problem.robots[num_to_id[robot_num]]
            robot.path = coords
            robot.current_position = coords[-1][:2]
            robot.active = False

        return {
            "solution": solution,
            "energy": pyo.value(builder.model.obj),
            "raw_response": results,
            "metadata": {
                "termination_condition": str(results.solver.termination_condition),
                "solver_config": self.to_dict(),
                # Single-entry list, not a per-window loop like QUBO's — ILP
                # solves the whole horizon in one shot. Wrapped in a list so
                # BenchmarkRunner.run_build()'s existing window_stats
                # aggregation (benchmark.py) picks it up unchanged.
                "window_stats": [builder.bfs_stats],
            },
        }
