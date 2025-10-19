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

    # Config file already returns dict for penalties; no new constructor needed
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
        # In case the QUBO is empty
        # It could happen in cases where the whole window gets pre-processed
        if len(Q) == 0:
            return Q
        # Extract values
        values = np.array(list(Q.values()))

        # Compute min/max
        max_val = np.max(np.abs(values))
        if max_val == 0:
            return Q

        # Scale all values to [-scale, scale]
        scale_factor = scale / max_val
        return {k: v * scale_factor for k, v in Q.items()}

    def decode_position(self, idx: int, problem) -> Tuple[int, int, int, int]:
        """
        Decode variable index to position, time, and robot number.

        Args:
            idx: Variable index
            problem: Problem instance

        Returns:
            Tuple of (i, j, t, robot_num) coordinates
        """
        if problem.get_format_type() == "graph":
            nodes_per_robot = len(problem.graph.nodes) * problem.T
            robot_num = idx // nodes_per_robot
            reduced_idx = idx % nodes_per_robot
            t = reduced_idx // len(problem.graph.nodes)
            graph_idx = reduced_idx % len(problem.graph.nodes)
            pos = problem.graph.get_node_position(graph_idx)
            return int(pos[0]), int(pos[1]), t, robot_num
        M = problem.grid.M
        N = problem.grid.N
        T = problem.T
        robot_num = idx // (M * N * T)
        reduced_idx = idx % (M * N * T)
        t = reduced_idx // (M * N)
        pos = reduced_idx % (M * N)
        i = pos // N
        j = pos % N
        return i, j, t, robot_num

    def decode_path(self, sample: Dict, problem, t_offset: int = 0) -> List[Tuple[Tuple[int, int, int], int]]:
        """
        Decode the binary sample into a path of ((i, j, t), robot_num) tuples.

        Args:
            sample: Binary solution as dict or list
            problem: Problem instance
            t_offset: Time offset for path segments

        Returns:
            List of ((i, j, t), robot_num) tuples representing the path
        """
        path = []

        # If sample is a list, process each element in order
        if isinstance(sample, list):
            if len(sample) == 1:
                sample = sample[0]
            else:  # Multiple segments (Time windows)
                t_offset_running = t_offset
                path = []
                for  index, s in enumerate(sample):
                    sub_path = self.decode_path(s, problem, t_offset=t_offset_running)
                    if sub_path:
                        max_t = max(x[0][2] for x in sub_path)
                        t_offset_running = max_t + 1
                    if path:
                        # Avoid duplicating the last position of the previous segment
                        print("Paths:", path[-1], sub_path[0])
                        if path and path[-1][0][:2] == sub_path[0][0][:2]:
                            # Updating time steps
                            # Else each would be off by one
                            sub_path = [((sb[0][0], sb[0][1], sb[0][2] - index), sb[1]) for sb in sub_path[1:]]
                    path.extend(sub_path)
                return path

        # If sample is a dict, process as before
        # This would be single window scenario
        if isinstance(sample, dict):
            qubo_type = problem.get_format_type()
            num_robots = problem.num_robots
            if qubo_type == "grid":
                M = problem.grid.M
                N = problem.grid.N
                T = problem.T
                total_vars = M * N * T * num_robots
            else:
                total_vars = len(problem.graph.nodes) * problem.T * num_robots
            for idx in range(total_vars):
                if sample.get(idx, 0) == 1:
                    i, j, t, robot_num = self.decode_position(idx, problem)
                    path.append(((i, j, t + t_offset), robot_num))
            return path

        return []

    def merge_path_segments(self, path: List[Tuple[Tuple[int, int, int], int]]) -> List[Tuple[Tuple[int, int, int], int]]:
        """
        Merge path segments and reindex time to be continuous.

        Args:
            path: List of ((i, j, t), robot_num) tuples

        Returns:
            Merged path with continuous time indexing
        """
        if not path:
            return []

        merged = [path[0]]
        for point in path[1:]:
            if (point[0][0], point[0][1], point[1]) == (merged[-1][0][0], merged[-1][0][1], merged[-1][1]):
                continue  # skip duplicate position for same robot
            merged.append(point)

        # Reindex t for each robot separately
        robot_paths = {}
        for (i, j, t), robot_num in merged:
            if robot_num not in robot_paths:
                robot_paths[robot_num] = []
            robot_paths[robot_num].append((i, j, t))
        
        # Reindex time for each robot
        result = []
        for robot_num, robot_path in robot_paths.items():
            for t, (i, j, _) in enumerate(robot_path):
                result.append(((i, j, t), robot_num))
        
        return result

    def get_combined_path(self, path: List[Tuple[Tuple[int, int, int], int]]) -> List[Tuple[int, int, int]]:
        """
        Get a combined path with all robots together, sorted by time.
        
        Args:
            path: List of ((i, j, t), robot_num) tuples
            
        Returns:
            List of (i, j, t) tuples sorted by time
        """
        if not path:
            return []
        
        # Sort by time, then by robot_num for consistent ordering
        sorted_path = sorted(path, key=lambda x: (x[0][2], x[1]))
        return [pos_time for (pos_time, robot_num) in sorted_path]

    def get_robot_paths(self, path: List[Tuple[Tuple[int, int, int], int]]) -> Dict[int, List[Tuple[int, int, int]]]:
        """
        Get individual paths for each robot.
        
        Args:
            path: List of ((i, j, t), robot_num) tuples
            
        Returns:
            Dictionary mapping robot_num to list of (i, j, t) tuples
        """
        robot_paths = {}
        
        for (i, j, t), robot_num in path:
            if robot_num not in robot_paths:
                robot_paths[robot_num] = []
            robot_paths[robot_num].append((i, j, t))
        
        # Sort each robot's path by time
        for robot_num in robot_paths:
            robot_paths[robot_num].sort(key=lambda x: x[2])
        
        return robot_paths

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

    def _handle_iteration_result(self, solution, fixed_vars, builder):
        """
        Handle the result of a QUBO iteration: reconstruct solution and 
        update problem.
        
        Args:
            solution: The solution dictionary or sample
            fixed_vars: Fixed variables from preprocessing
            builder: QUBOBuilder instance
            
        Returns:
            tuple: (reconstructed_solution, success_flag)
        """
        # Reconstruct full solution
        full_sol = builder.reconstruct_solution(
            solution,
            fixed_vars,
            total_vars=builder.initial_num_vars
        )
        
        # Update problem for next iteration
        try:
            path = self.decode_path(full_sol, builder.problem)
            last_pos = path[-1]
            print("Decoded path:", path)
            print("Last position:", last_pos)
            builder.update_problem(last_pos[0][:2], path)
            return full_sol, True
        except Exception as e:
            print(f"Warning: Could not decode path: {e}")
            return full_sol, False
