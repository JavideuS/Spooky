# Solver (Quantum annealing)
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from .base_solver import BaseSolver


class DWaveSolver(BaseSolver):
    def __init__(self, normalize_scale=0, num_reads=10, **kwargs):
        super().__init__(backend="dwave", normalize_scale=normalize_scale,
                         num_reads=num_reads, **kwargs)

    def solve_qubo(self, builder):
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

            # Get best solution
            # This simply extract the solution with the lowest energy (Theoretically the best solution)
            best_sample.append(response.first.sample)
            best_energy.append(response.first.energy)
            last_pos = self.decode_path(response.first.sample, builder.problem)[-1]
            builder.update_problem(last_pos[:2])

        return {
            'solution': best_sample,
            'energy': best_energy,
            # 'success': is_solution_valid(best_sample, M, N, T, s_i, s_j, e_i, e_j),
            'raw_response': response,
        }


# Backward compatibility - keep the old class name
class QUBOSolver(DWaveSolver):
    """Backward compatibility alias for DWaveSolver"""
    pass
