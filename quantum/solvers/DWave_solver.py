# Solver (Quantum annealing)
from tracemalloc import start
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from .base_solver import BaseSolver


class DWaveSolver(BaseSolver):
    def __init__(self, normalize_scale=0, num_reads=10, verbose_level=2, **kwargs):
        super().__init__(solver="dwave", normalize_scale=normalize_scale,
                         num_reads=num_reads, verbose_level=verbose_level, **kwargs)

    def solve_qubo(self, builder, opt):
        best_sample = []
        best_energy = []
        response = None

        while (builder.total_t) > (builder.current_T):
            Q = builder.Q
            if self.norm_scale != 0:
                Q = self.normalize_qubo(builder.Q, self.norm_scale)
            self.logger.standard("Start position:", builder.problem.start, "Iteration:", builder.iter)
            bqm = BinaryQuadraticModel.from_qubo(Q)
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=self.num_reads)

            first = response.first
            # sample_dict = dict(first.sample)  # OrderedDict → dict
            # print("Sample:", self.decode_path(sample_dict, builder.problem))
            best_sample.append(first.sample)
            best_energy.append(response.first.energy)
            last_pos = self.decode_path(first.sample, builder.problem)[-1]
            builder.update_problem(last_pos[:2])

        return {
            'solution': best_sample,
            'energy': best_energy,
            # 'success': is_solution_valid(best_sample, M, N, T, s_i, s_j, e_i, e_j),
            'raw_response': response,
        }
    
    def solve_qubo_smart(self, builder, opt):
        best_sample = []
        best_energy = []
        window_stats = []  # Track per-window variable reduction stats
        response = None
        correction_count = 0  # Track consecutive correction attempts for current window
        import time as timing  # For profiling

        while (builder.total_t) > (builder.current_T):
            # Ensure QUBO is built for the start of the window
            # If the user forgot to call builder.build(), we do it here
            if not builder.Q:
                self.logger.debug("QUBO has not been built, calling builder.build()")
                builder.build()

            # TERMINATION CHECK: If all robots are inactive, stop solving
            active_robots = [r for r in builder.problem.robots.values() if r.active]
            if not active_robots:
                self.logger.standard("✅ All robots reached goal or inactive. Stopping solver.")
                break


            window_start = timing.time()
            
            # print("Pre Num wires", builder.get_num_wires())
            if self.norm_scale != 0:
                # Get initial variable count BEFORE reduction
                initial_vars = builder.get_num_wires()
                
                t0 = timing.time()
                fixed_vars = builder.get_fixed_variables()
                t1 = timing.time()
                # Enable logging for initial reduction so BFS recalculation can unfix variables
                builder.Q, const_offset, initial_reduction_log = builder.reduce_qubo(fixed_vars, log_reductions=True)
                t2 = timing.time()
                # Pass initial reduction log to diagonal reduction so it can unfix variables if needed
                diag_fixed = builder.reduce_diag_fixed_vars_iterative(initial_reduction_log=initial_reduction_log)
                t3 = timing.time()
                fixed_vars.update(diag_fixed)
                
                self.logger.debug(f"⏱️ get_fixed_vars: {(t1-t0)*1000:.1f}ms, reduce_qubo: {(t2-t1)*1000:.1f}ms, diag_reduce: {(t3-t2)*1000:.1f}ms")
                
                # Count total reduced variables (BFS fixed + diagonal fixed)
                vars_reduced = len(fixed_vars)
                
                # Get final variable count AFTER reduction
                final_vars = builder.get_num_wires()
                # print(final_vars)
                reduction_ratio = vars_reduced / initial_vars if initial_vars > 0 else 0
                
                # Store per-window stats
                window_stats.append({
                    "window": builder.iter,
                    "initial_variables": initial_vars,
                    "variables_reduced": vars_reduced,
                    "final_variables": final_vars,
                    "reduction_ratio": round(reduction_ratio, 4)
                })
                
                self.logger.standard(
                    f"Window {builder.iter}: {initial_vars} → {final_vars} vars "
                    f"(reduced {vars_reduced}, {reduction_ratio:.1%})"
                )
                
                # FAST PATH: Skip sampler if QUBO is fully pre-processed
                if len(builder.Q) == 0 or final_vars == 0:
                    self.logger.standard(f"⚡ Window {builder.iter} fully pre-processed, skipping solver")
                    t4 = timing.time()
                    full_sol, invalid_moves = self._handle_iteration_result({}, fixed_vars, builder)
                    t5 = timing.time()
                    self.logger.debug(f"⏱️ _handle_iteration_result: {(t5-t4)*1000:.1f}ms, total window: {(t5-window_start)*1000:.1f}ms")
                    best_sample.append(full_sol)
                    best_energy.append(0.0)
                    continue
                
                # print(fixed_vars)
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)
            # print(fixed_vars)
            self.logger.standard("Num wires", builder.get_num_wires())
            for _, robot_id in enumerate(builder.problem.robots):
                start_pos = builder.problem.robots[robot_id].current_position
                self.logger.standard("Start position:", start_pos, "Iteration:", builder.iter)
            bqm = BinaryQuadraticModel.from_qubo(builder.Q)
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=self.num_reads)

            first = response.first
            # print("Sample:", self.decode_path(sample_dict, builder.problem))
            full_sol, invalid_moves = self._handle_iteration_result(first.sample, fixed_vars, builder)
            best_sample.append(full_sol)
            best_energy.append(response.first.energy)
            
            # Check if correction is needed due to invalid moves
            if invalid_moves:
                correction_count += 1
                self.logger.standard(f"🔄 Correction attempt {correction_count}/{self.max_corrections} for current window")
                
                if correction_count >= self.max_corrections:
                    self.logger.minimal(f"⚠️  Max corrections ({self.max_corrections}) exceeded at t={builder.current_T}. "
                                       f"Keeping last result (invalid moves for robots {list(invalid_moves.keys())}).")
                    
                    # CRITICAL: Force builder to advance to next window to avoid infinite loop
                    # Decode the path from the invalid solution and update the problem
                    path = self.decode_path(full_sol, builder.problem, t_offset=builder.current_T)
                    robot_paths = self.get_robot_paths(path)
                    robot_paths = self._resolve_duplicate_timesteps(robot_paths, builder.problem)
                    # Note: We skip _resolve_invalid_moves since we already know there are invalid moves
                    # and we want to accept them to move forward
                    builder.update_problem(robot_paths)
                    
                    correction_count = 0  # Reset for next window
                else:
                    # Rebuild QUBO from scratch for retry attempt
                    # This is critical because the QUBO was reduced during the failed attempt
                    # and we need a fresh QUBO to try again
                    self.logger.debug(f"Rebuilding QUBO for retry attempt {correction_count}")
                    builder.build()
            else:
                # Successful window - reset correction counter
                correction_count = 0

        # Build final solution from stored robot paths (this is the correct solution)
        final_solution = self.build_solution_from_robot_paths(builder.problem)
        
        return {
            'solution': final_solution,  # Use solution built from robot paths
            'energy': best_energy,
            'raw_response': response,
            'metadata': {
                'window_stats': window_stats,  # Per-window variable reduction stats
                'num_robots': builder.problem.num_robots,
                'total_variables': builder.initial_num_vars,
                'fixed_variables': len(fixed_vars) if 'fixed_vars' in dir() else 0,
                'constant_offset': const_offset if 'const_offset' in dir() else 0,
                'solver_config': self.to_dict(),
                'penalties': builder.penalties,
                # 'window_solutions': best_sample  # Keep window solutions for debugging
            }
        }

