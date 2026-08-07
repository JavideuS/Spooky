# Solver (Quantum annealing)
from dimod import BinaryQuadraticModel

# Optimized version is neal AnnealingSampler not dimod one (C++ based)
from neal import SimulatedAnnealingSampler
from .base_solver import BaseSolver


class DWaveSolver(BaseSolver):
    def __init__(
        self, normalize_scale=0, num_reads=10, verbose_level=2, seed=None, **kwargs
    ):
        """
        Args:
            seed: Random seed forwarded to neal.SimulatedAnnealingSampler.sample().
                None (default) leaves annealing non-deterministic run-to-run;
                set for reproducible sweeps/benchmarks.
        """
        super().__init__(
            solver="dwave",
            normalize_scale=normalize_scale,
            num_reads=num_reads,
            verbose_level=verbose_level,
            seed=seed,
            **kwargs,
        )
        self.seed = seed

    def solve(self, builder, optimization=False, preprocess=True):
        """
        Solve QUBO using simulated annealing.

        Args:
            builder: QUBOBuilder instance
            optimization: Accepted for interface compatibility; unused by DWave/SA.
            preprocess: When True (default), applies BFS variable reduction,
                diagonal pruning, correction loop, and window stats tracking.
                When False, runs a simple loop with no preprocessing.

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        best_sample = []
        best_energy = []
        window_stats = []
        forced_collisions = []
        response = None
        correction_count = 0
        import time as timing

        if not preprocess:
            # Simple loop — no variable reduction, no correction retries
            while (builder.total_t) > (builder.current_T):
                Q = builder.Q
                if self.norm_scale != 0:
                    Q = self.normalize_qubo(builder.Q, self.norm_scale)
                self.logger.standard(
                    "Start position:", builder.problem.start, "Iteration:", builder.iter
                )
                bqm = BinaryQuadraticModel.from_qubo(Q)
                sampler = SimulatedAnnealingSampler()
                response = sampler.sample(bqm, num_reads=self.num_reads, seed=self.seed)
                first = response.first
                best_sample.append(first.sample)
                best_energy.append(response.first.energy)
                last_pos = self.decode_path(first.sample, builder.problem)[-1]
                builder.update_problem(last_pos[:2])

            return {
                "solution": best_sample,
                "energy": best_energy,
                "raw_response": response,
            }

        # preprocess=True: full pipeline with variable reduction and correction loop
        while (builder.total_t) > (builder.current_T):
            active_robots = [r for r in builder.problem.robots.values() if r.active]
            if not active_robots:
                self.logger.standard(
                    "✅ All robots reached goal or inactive. Stopping solver."
                )
                break

            window_start = timing.time()
            fixed_vars, window_stat, is_preprocessed, window_forced_collisions = (
                self._prepare_window(builder)
            )
            window_stats.append(window_stat)
            forced_collisions.extend(window_forced_collisions)

            if is_preprocessed:
                self.logger.standard(
                    f"⚡ Window {builder.iter} fully pre-processed, skipping solver"
                )
                t_fast = timing.time()
                full_sol, invalid_moves = self._handle_iteration_result(
                    {}, fixed_vars, builder
                )
                self.logger.debug(
                    f"⏱️ _handle_iteration_result: {(timing.time() - t_fast) * 1000:.1f}ms, "
                    f"total window: {(timing.time() - window_start) * 1000:.1f}ms"
                )
                best_sample.append(full_sol)
                best_energy.append(0.0)
                continue

            if self.norm_scale != 0:
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)

            self.logger.standard("Num wires", builder.get_num_wires())
            for _, robot_id in enumerate(builder.problem.robots):
                start_pos = builder.problem.robots[robot_id].current_position
                self.logger.standard(
                    "Start position:", start_pos, "Iteration:", builder.iter
                )

            bqm = BinaryQuadraticModel.from_qubo(builder.Q)
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=self.num_reads, seed=self.seed)

            first = response.first
            full_sol, invalid_moves = self._handle_iteration_result(
                first.sample, fixed_vars, builder
            )
            best_sample.append(full_sol)
            best_energy.append(response.first.energy)

            if invalid_moves:
                correction_count += 1
                self.logger.standard(
                    f"🔄 Correction attempt {correction_count}/{self.max_corrections} for current window"
                )

                if correction_count >= self.max_corrections:
                    self.logger.minimal(
                        f"⚠️  Max corrections ({self.max_corrections}) exceeded at t={builder.current_T}. "
                        f"Keeping last result (invalid moves for robots {list(invalid_moves.keys())})."
                    )
                    path = self.decode_path(
                        full_sol, builder.problem, t_offset=builder.current_T
                    )
                    robot_paths = self.get_robot_paths(path)
                    robot_paths = self._resolve_duplicate_timesteps(
                        robot_paths, builder.problem
                    )
                    builder.update_problem(robot_paths)
                    correction_count = 0
                # else: next loop iteration calls _prepare_window to rebuild from scratch
            else:
                correction_count = 0

        final_solution = self.build_solution_from_robot_paths(builder.problem)

        return {
            "solution": final_solution,
            "energy": best_energy,
            "raw_response": response,
            "metadata": {
                "window_stats": window_stats,
                "forced_collisions": forced_collisions,
                "num_robots": builder.problem.num_robots,
                "total_variables": builder.initial_num_vars,
                "fixed_variables": len(fixed_vars) if "fixed_vars" in dir() else 0,
                "solver_config": self.to_dict(),
                "penalties": builder.penalties,
            },
        }
