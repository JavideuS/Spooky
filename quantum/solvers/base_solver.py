from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple


class BaseSolver(ABC):
    """
    Abstract base class for all quantum solvers.
    Defines the common interface that all solvers must implement.
    """

    def __init__(self, backend: str, normalize_scale: float = 0,
                 num_reads: int = 10, **kwargs):
        """
        Initialize the base solver.

        Args:
            backend: Name of the backend (e.g., "dwave", "pennylane", "qiskit")
            normalize_scale: Scale factor for QUBO normalization
            num_reads: Number of reads/samples to take
            **kwargs: Additional backend-specific parameters
        """
        self.backend = backend
        self.norm_scale = normalize_scale
        self.num_reads = num_reads
        self.name = f"{self.backend}_reads{num_reads}"
        self._backend_params = kwargs

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Create a solver instance from a configuration dictionary.

        Args:
            config: Configuration dictionary containing solver parameters

        Returns:
            Solver instance
        """
        norm_scale = config.get("normalization_scale", 0)
        num_reads = config.get("num_reads", 15)

        # Extract backend-specific parameters
        backend_params = {
            k: v for k, v in config.items() 
            if k not in ["backend", "normalization_scale", "num_reads"]
        }

        # Each class expects to run its own backend
        return cls(normalize_scale=norm_scale,
                   num_reads=num_reads, **backend_params)

    def normalize_qubo(self, Q: Dict, scale: float = 1.0) -> Dict:
        """
        Normalize QUBO coefficients to a specified scale.

        Args:
            Q: QUBO dictionary
            scale: Target scale for normalization

        Returns:
            Normalized QUBO dictionary
        """
        # Extract values
        values = np.array(list(Q.values()))

        # Compute min/max
        max_val = np.max(np.abs(values))
        if max_val == 0:
            return Q

        # Scale all values to [-scale, scale]
        scale_factor = scale / max_val
        return {k: v * scale_factor for k, v in Q.items()}

    def decode_position(self, idx: int, problem) -> Tuple[int, int, int]:
        """
        Decode variable index to position and time.

        Args:
            idx: Variable index
            problem: Problem instance

        Returns:
            Tuple of (i, j, t) coordinates
        """
        M = problem.grid.M
        N = problem.grid.N
        t = idx // (M * N)
        pos = idx % (M * N)
        i = pos // N
        j = pos % N
        return i, j, t

    def decode_path(self, sample: Dict, problem, t_offset: int = 0) -> List[Tuple[int, int, int]]:
        """
        Decode the binary sample into a path of (i, j, t) tuples.

        Args:
            sample: Binary solution as dict or list
            problem: Problem instance
            t_offset: Time offset for path segments

        Returns:
            List of (i, j, t) tuples representing the path
        """
        path = []

        # If sample is a list, process each element in order
        if isinstance(sample, list):
            if len(sample) == 1:
                sample = sample[0]
            else:
                t_offset_running = t_offset
                path = []
                for  index, s in enumerate(sample):
                    sub_path = self.decode_path(s, problem, t_offset=t_offset_running)
                    if sub_path:
                        max_t = max(x[2] for x in sub_path)
                        t_offset_running = max_t + 1
                    if path:
                        # Avoid duplicating the last position of the previous segment
                        print("Paths:", path[-1], sub_path[0])
                        if path and path[-1][:2] == sub_path[0][:2]:
                            # Updating time steps
                            # Else each would be off by one
                            sub_path = [(sb[0], sb[1], sb[2] - index) for sb in sub_path[1:]]
                    path.extend(sub_path)
                return path

        # If sample is a dict, process as before
        # This would be single window scenario
        if isinstance(sample, dict):
            M = problem.grid.M
            N = problem.grid.N
            T = problem.T
            for idx in range(M * N * T):
                if sample.get(idx, 0) == 1:
                    i, j, t = self.decode_position(idx, problem)
                    path.append((i, j, t + t_offset))
            return path

        return []

    def merge_path_segments(self, path: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """
        Merge path segments and reindex time to be continuous.

        Args:
            path: List of (i, j, t) tuples

        Returns:
            Merged path with continuous time indexing
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

    @abstractmethod
    def solve_qubo(self, builder) -> Dict[str, Any]:
        """
        Solve the QUBO problem.

        Args:
            builder: QUBOBuilder instance

        Returns:
            Dictionary containing solution, energy, and raw response
        """
        pass

    def total_energy(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the total energy of all windows in the solution.

        Args:
            solution: Solution dictionary

        Returns:
            Total energy
        """
        return np.sum(solution["energy"])

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert solver parameters to dictionary.

        Returns:
            Dictionary representation of solver parameters
        """
        result = {
            "backend": self.backend,
            "normalization_scale": self.norm_scale,
            "num_reads": self.num_reads,
        }
        result.update(self._backend_params)
        return result

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get backend-specific information.

        Returns:
            Dictionary with backend information
        """
        return {
            "backend": self.backend,
            "name": self.name,
            "parameters": self._backend_params
        }
