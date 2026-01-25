# Solver (Quantum annealing)
from tracemalloc import start
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from .base_solver import BaseSolver


class DWaveSolver(BaseSolver):
    def __init__(self, normalize_scale=0, num_reads=10, **kwargs):
        super().__init__(solver="dwave", normalize_scale=normalize_scale,
                         num_reads=num_reads, **kwargs)

    def solve_qubo(self, builder, opt):
        best_sample = []
        best_energy = []
        response = None

        while (builder.total_t) > (builder.current_T):
            Q = builder.Q
            if self.norm_scale != 0:
                Q = self.normalize_qubo(builder.Q, self.norm_scale)
            print("Start position:", builder.problem.start, "Iteration:", builder.iter)
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
        response = None

        while (builder.total_t) > (builder.current_T):
            # print("Pre Num wires", builder.get_num_wires())
            if self.norm_scale != 0:
                fixed_vars = builder.get_fixed_variables()
                builder.Q, const_offset, log = builder.reduce_qubo(fixed_vars)
                diag_fixed = builder.reduce_diag_fixed_vars_iterative(log)
                fixed_vars.update(diag_fixed)
                # print(fixed_vars)
                builder.Q = self.normalize_qubo(builder.Q, self.norm_scale)
            # print(fixed_vars)
            print("Num wires", builder.get_num_wires())
            for _, robot_id in enumerate(builder.problem.robots):
                start_pos = builder.problem.robots[robot_id].current_position
                print("Start position:", start_pos, "Iteration:", builder.iter)
            bqm = BinaryQuadraticModel.from_qubo(builder.Q)
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=self.num_reads)

            first = response.first
            # print("Sample:", self.decode_path(sample_dict, builder.problem))
            full_sol, invalid_moves = self._handle_iteration_result(first.sample, fixed_vars, builder)
            best_sample.append(full_sol)
            best_energy.append(response.first.energy)

        # Build final solution from stored robot paths (this is the correct solution)
        final_solution = self.build_solution_from_robot_paths(builder.problem)
        
        return {
            'solution': final_solution,  # Use solution built from robot paths
            'energy': best_energy,
            'raw_response': response,
            'metadata': {
                'num_robots': builder.problem.num_robots,
                'total_variables': builder.initial_num_vars,
                'fixed_variables': len(fixed_vars),
                'constant_offset': const_offset,
                'solver_config': self.to_dict(),
                'penalties': builder.penalties,
                # 'window_solutions': best_sample  # Keep window solutions for debugging
            }
        }
