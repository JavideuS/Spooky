from typing import Dict, Any, Type
from .base_solver import BaseSolver
from .DWave_solver import DWaveSolver
from .Pennylane_solver import PennylaneSolver


class SolverFactory:
    """
    Factory class for creating quantum solvers.
    Supports dynamic backend switching and configuration-based solver creation.
    """
    
    # Registry of available solvers
    _solvers: Dict[str, Type[BaseSolver]] = {
        "dwave": DWaveSolver,
        "pennylane": PennylaneSolver,
    }
    
    @classmethod
    def register_solver(cls, backend_name: str, solver_class: Type[BaseSolver]):
        """
        Register a new solver backend.
        
        Args:
            backend_name: Name of the backend
            solver_class: Solver class that inherits from BaseSolver
        """
        cls._solvers[backend_name] = solver_class
    
    @classmethod
    def get_available_backends(cls) -> list:
        """
        Get list of available backend names.
        
        Returns:
            List of available backend names
        """
        return list(cls._solvers.keys())
    
    @classmethod
    def create_solver(cls, backend: str, **kwargs) -> BaseSolver:
        """
        Create a solver instance for the specified backend.
        
        Args:
            backend: Name of the backend
            **kwargs: Solver-specific parameters
            
        Returns:
            Solver instance
            
        Raises:
            ValueError: If backend is not supported
        """
        if backend not in cls._solvers:
            available = ", ".join(cls.get_available_backends())
            raise ValueError(f"Backend '{backend}' not supported. "
                          f"Available backends: {available}")
        
        solver_class = cls._solvers[backend]
        return solver_class(**kwargs)
    
    @classmethod
    def create_solver_from_config(cls, config: Dict[str, Any]) -> BaseSolver:
        """
        Create a solver instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary with solver parameters
            
        Returns:
            Solver instance
        """
        backend = config.get("backend", "dwave")
        
        if backend not in cls._solvers:
            available = ", ".join(cls.get_available_backends())
            raise ValueError(f"Backend '{backend}' not supported. "
                          f"Available backends: {available}")
        
        solver_class = cls._solvers[backend]
        return solver_class.from_config(config)
    
    @classmethod
    def switch_backend(cls, solver: BaseSolver, new_backend: str, 
                      **kwargs) -> BaseSolver:
        """
        Switch a solver to a different backend while preserving common parameters.
        
        Args:
            solver: Current solver instance
            new_backend: New backend name
            **kwargs: Additional parameters for the new backend
            
        Returns:
            New solver instance with the specified backend
        """
        # Extract common parameters from current solver
        common_params = {
            "normalize_scale": solver.norm_scale,
            "num_reads": solver.num_reads,
        }
        
        # Update with new parameters
        common_params.update(kwargs)
        
        return cls.create_solver(new_backend, **common_params)


class DynamicSolver:
    """
    A wrapper class that allows dynamic backend switching during runtime.
    """
    
    def __init__(self, initial_backend: str = "dwave", **kwargs):
        """
        Initialize with a specific backend.
        
        Args:
            initial_backend: Initial backend to use
            **kwargs: Solver parameters
        """
        self.current_backend = initial_backend
        self.solver = SolverFactory.create_solver(initial_backend, **kwargs)
    
    def switch_backend(self, new_backend: str, **kwargs):
        """
        Switch to a different backend.
        
        Args:
            new_backend: New backend name
            **kwargs: Additional parameters for the new backend
        """
        self.solver = SolverFactory.switch_backend(
            self.solver, new_backend, **kwargs
        )
        self.current_backend = new_backend
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the current backend.
        
        Returns:
            Dictionary with backend information
        """
        return self.solver.get_backend_info()
    
    def solve_qubo(self, builder):
        """
        Solve QUBO using the current backend.
        
        Args:
            builder: QUBOBuilder instance
            
        Returns:
            Solution dictionary
        """
        return self.solver.solve_qubo(builder)
    
    def decode_path(self, sample, problem, t_offset=0):
        """
        Decode path using the current solver.
        
        Args:
            sample: Binary solution
            problem: Problem instance
            t_offset: Time offset
            
        Returns:
            Decoded path
        """
        return self.solver.decode_path(sample, problem, t_offset)
    
    def total_energy(self, solution):
        """
        Calculate total energy using the current solver.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Total energy
        """
        return self.solver.total_energy(solution)
    
    def to_dict(self):
        """
        Get solver parameters as dictionary.
        
        Returns:
            Dictionary representation
        """
        return self.solver.to_dict() 