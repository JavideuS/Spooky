# Solver (Quantum annealing)
from dimod import BinaryQuadraticModel, SimulatedAnnealingSampler
from collections import Counter
import numpy as np


class QUBOSolver:
    def __init__(self, normalize_scale=0, num_reads=10):
        self.norm_scale = normalize_scale
        self.num_reads = num_reads
        self.backend = "dwave_qa"
        self.name = f"{self.backend}_reads{num_reads}"

    @classmethod
    def from_config(cls, config):
        """
        Create a QUBOSolver instance from a configuration dictionary.
        It receives solver section from config file
        and extracts solver parameters.
        """
        norm_scale = config.get("normalization_scale", 0)
        num_reads = config.get("num_reads", 10)
        # backend = config.get("backend", "dwave_qa")
        return cls(normalize_scale=norm_scale, num_reads=num_reads)

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
    
    # Helper function to get variable index
    def decode_position(self, idx, problem):
        M = problem.grid.M
        N = problem.grid.N
        t = idx // (M * N)
        pos = idx % (M * N)
        i = pos // N
        j = pos % N
        return i, j, t
    
    def decode_path(self, sample, problem):
        """
        Decode the binary sample into a path of (i, j, t) tuples.
        """
        path = []
        M = problem.grid.M
        N = problem.grid.N
        T = problem.T

        for idx in range(M * N * T):
            if sample.get(idx, 0) == 1:
                i, j, t = self.decode_position(idx, problem)
                path.append((i, j, t))

        return sorted(path, key=lambda x: x[2])

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
            'raw_response': response,
        }
    
    def get_path(self, sample, problem):
        """
        Get the path from the sample.
        """
        return self.decode_path(sample, problem)
    
    def to_dict(self):
        """
        Convert the solver parameters to a dictionary.
        """
        return {
            "normalization_scale": self.norm_scale,
            "num_reads": self.num_reads,
        }

