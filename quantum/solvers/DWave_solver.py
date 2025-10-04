# Solver (Quantum annealing)
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from .base_solver import BaseSolver


class DWaveSolver(BaseSolver):
    def __init__(self, normalize_scale=0, num_reads=10, **kwargs):
        super().__init__(backend="dwave", normalize_scale=normalize_scale,
                         num_reads=num_reads, **kwargs)

    def solve_qubo(self, builder, opt):
        best_sample = []
        best_energy = []
        response = None

        while (builder.total_t) > (builder.iter * builder.t_max):
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

        while (builder.total_t) > (builder.iter * builder.t_max):
            Q = builder.Q
            # print("Pre Num wires", builder.get_num_wires())
            if self.norm_scale != 0:
                fixed_vars = builder.get_fixed_variables()
                builder.Q, offset = builder.reduce_qubo(fixed_vars)
                # builder.Q = Q
                # numerical_fixes = builder.diag_fixed_vars()
                # print(numerical_fixes)
                # Q, offset2 = builder.reduce_qubo(numerical_fixes)
                # fixed_vars.update(numerical_fixes)
                diag_fixed = builder.reduce_diag_fixed_vars_iterative()
                fixed_vars.update(diag_fixed)
                Q = self.normalize_qubo(builder.Q, self.norm_scale)
            # builder.Q = Q
            # print("Num wires", builder.get_num_wires())
            print("Start position:", builder.problem.start, "Iteration:", builder.iter)
            bqm = BinaryQuadraticModel.from_qubo(Q)
            sampler = SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=self.num_reads)

            first = response.first
            # sample_dict = dict(first.sample)  # OrderedDict → dict
            # print("Sample:", self.decode_path(sample_dict, builder.problem))
            full_sol = builder.reconstruct_solution(
                first.sample,
                fixed_vars,
                total_vars=builder.initial_num_vars
            )
            best_sample.append(full_sol)
            best_energy.append(response.first.energy)
            last_pos = self.decode_path(full_sol, builder.problem)[-1]
            builder.update_problem(last_pos[:2])

        return {
            'solution': best_sample,
            'energy': best_energy,
            # 'success': is_solution_valid(best_sample, M, N, T, s_i, s_j, e_i, e_j),
            'raw_response': response,
        }
