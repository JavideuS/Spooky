from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, List, Tuple
from quantum.utils.paths import decode_position

class BaseSolver(ABC):
    """
    Abstract base class for all quantum solvers.
    Defines the common interface that all solvers must implement.
    """

    def __init__(self, solver: str, normalize_scale: float = 0,
                 num_reads: int = 10, **kwargs):
        """
        Initialize the base solver.

        Args:
            solver: Name of the solver (e.g., "dwave", "pennylane", "qiskit")
            normalize_scale: Scale factor for QUBO normalization
            num_reads: Number of reads/samples to take
            **kwargs: Additional solver-specific parameters
        """
        self.solver = solver
        self.norm_scale = normalize_scale
        self.num_reads = num_reads
        self.name = f"{self.solver}_reads{num_reads}"
        self._solver_params = kwargs

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

        # Extract solver-specific parameters
        solver_params = {
            k: v for k, v in config.items() 
            if k not in ["solver", "normalization_scale", "num_reads"]
        }

        # Each class expects to run its own solver
        return cls(normalize_scale=norm_scale,
                   num_reads=num_reads, **solver_params)

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

    def decode_path(self, sample: Dict, problem, t_offset: int = 0) -> List[Tuple[Tuple[int, int, int], int]]:
        """
        Decode the binary sample into a path of ((i, j, t), robot_num) tuples.
        Merges multiple time-window samples while ensuring continuity per robot.
        """
        path = []

        # Handle multiple segments (list of samples)
        if isinstance(sample, list):
            if len(sample) == 1:
                sample = sample[0]
            else:
                t_offset_running = t_offset
                path = []
                # Store last position per robot
                last_positions = {}

                for index, s in enumerate(sample):
                    sub_path = self.decode_path(s, problem, t_offset=t_offset_running)
                    if not sub_path:
                        continue

                    # Organize by robot for continuity checking
                    sub_robot_paths = self.get_robot_paths(sub_path)

                    # For each robot, check if we can clip the start
                    for robot_num, robot_path in sub_robot_paths.items():
                        if robot_num in last_positions:
                            last_pos = last_positions[robot_num]
                            first_pos = robot_path[0][:2]
                            if last_pos == first_pos:
                                # Same position continuity → remove first from subpath
                                sub_robot_paths[robot_num] = robot_path[1:]

                                # Also adjust time for continuity
                                sub_robot_paths[robot_num] = [
                                    (i, j, t - 1) for (i, j, t) in sub_robot_paths[robot_num]
                                ]
                    
                    # We have been working with dict num_robot: path
                    # So we flatten sub_robot_paths back to list of ((i,j,t), robot_num)
                    merged_sub_path = [((i, j, t), r) for r, coords in sub_robot_paths.items() for (i, j, t) in coords]

                    # Update last_positions
                    for (i, j, t), r in merged_sub_path:
                        last_positions[r] = (i, j)

                    # Update running time offset
                    if merged_sub_path:
                        max_t = max(x[0][2] for x in merged_sub_path)
                        t_offset_running = max_t + 1

                    path.extend(merged_sub_path)

                return path

        # Handle single dict sample
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
                    i, j, t, robot_num = decode_position(idx, problem)
                    path.append(((i, j, t + t_offset), robot_num))

            return path

        return []

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
            "solver": self.solver,
            "normalization_scale": self.norm_scale,
            "num_reads": self.num_reads,
        }
        result.update(self._solver_params)
        return result

    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get solver-specific information.

        Returns:
            Dictionary with solver information
        """
        return {
            "solver": self.solver,
            "name": self.name,
            "parameters": self._solver_params
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
        
        path = self.decode_path(full_sol, builder.problem)
        robot_paths = self.get_robot_paths(path)
        # print("Decoded path:", path)
        print("Robots paths", robot_paths)
        builder.update_problem(robot_paths)
        return full_sol
