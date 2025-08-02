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
    
    def decode_path(self, sample, problem, t_offset=0):
        """
        Decode the binary sample into a path of (i, j, t) tuples.
        Accepts sample as a dict or a list (ordered solution).
        If sample is a list, concatenates the decoded path for each element,
        and offsets t so that it is continuous.
        """
        path = []
        # If sample is a list, process each element in order
        if isinstance(sample, list):
            # If it's a one-element list, treat the element as the sample
            if len(sample) == 1:
                sample = sample[0]
                # fall through to dict case
            else:
                t_offset_running = t_offset
                path = []
                for s in sample:
                    sub_path = self.decode_path(s, problem, t_offset=t_offset_running)
                    if sub_path:
                        max_t = max(x[2] for x in sub_path)
                        t_offset_running = max_t + 1
                    path.extend(sub_path)
                return path
        # If sample is a dict, process as before
        if isinstance(sample, dict):
            M = problem.grid.M
            N = problem.grid.N
            T = problem.T
            for idx in range(M * N * T):
                if sample.get(idx, 0) == 1:
                    i, j, t = self.decode_position(idx, problem)
                    path.append((i, j, t + t_offset))
            return path
        # If sample is neither dict nor list, return empty path
        return []

    def merge_path_segments(self, path):
        """
        Given a path as a list of (i, j, t), remove duplicate consecutive positions
        (where the first of a segment matches the last of the previous), and
        reindex t to be continuous from 0.
        """
        if not path:
            return []
        merged = [path[0]]
        for point in path[1:]:
            if (point[0], point[1]) == (merged[-1][0], merged[-1][1]):
                continue  # skip duplicate position
            merged.append(point)
        # Reindex t
        merged = [(i, j, t) for t, (i, j, _) in enumerate(merged)]
        return merged

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
    
    def to_dict(self):
        """
        Convert the solver parameters to a dictionary.
        """
        return {
            "normalization_scale": self.norm_scale,
            "num_reads": self.num_reads,
        }

    def total_energy(self, solution):
        "Calculates the total energy of all windows in the solution."
        return np.sum(solution["energy"])
