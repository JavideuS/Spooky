# Solver (Quantum annealing)
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from collections import Counter
import numpy as np


class QUBOSolver:
    def __init__(self, normalize_scale=0, num_reads=10):
        self.norm_scale = normalize_scale
        self.num_reads = num_reads

    def normalize_qubo(self, Q, scale=1.0):
        # Extract values
        values = np.array(list(Q.values()))

        # Compute min/max
        max_val = np.max(np.abs(values))
        if max_val == 0:
            return Q

        # Scale all values to [-scale, scale]
        scale_factor = scale / max_val
        return {k: v * scale_factor for k, v in Q.items()}

    def solve_qubo(self, Q):
        if self.norm_scale != 0:
            Q = self.normalize_qubo(Q, self.norm_scale)

        bqm = BinaryQuadraticModel.from_qubo(Q)
        sampler = SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=self.num_reads)

        # Get best solution
        # This simply extract the solution with the lowest energy (Theoretically the best solution)
        best_sample = response.first.sample
        best_energy = response.first.energy

        # top_solutions = sorted(response.data(), key=lambda x: x.energy)[:4]
        # samples = [tuple(sorted(sol.sample.items())) for sol in top_solutions]
        # counter = Counter(samples)
        # best_sample = dict(counter.most_common(1)[0][0])
        # best_energy = min(sol.energy for sol in top_solutions)

        return {
            'solution': best_sample,
            'energy': best_energy,
            # 'success': is_solution_valid(best_sample, M, N, T, s_i, s_j, e_i, e_j),
            'raw_response': response
        }

