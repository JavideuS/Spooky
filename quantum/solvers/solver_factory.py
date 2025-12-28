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
    def register_solver(cls, solver_name: str, solver_class: Type[BaseSolver]):
        """
        Register a new solver backend.
        
        Args:
            solver_name: Name of the solver
            solver_class: Solver class that inherits from BaseSolver
        """
        cls._solvers[solver_name] = solver_class
    
    @classmethod
    def get_available_solvers(cls) -> list:
        """
        Get list of available solver names.
        
        Returns:
            List of available solver names
        """
        return list(cls._solvers.keys())
    
    @classmethod
    def create_solver(cls, solver: str, **kwargs) -> BaseSolver:
        """
        Create a solver instance for the specified solver.
        
        Args:
            solver: Name of the solver
            **kwargs: Solver-specific parameters
            
        Returns:
            Solver instance
            
        Raises:
            ValueError: If solver is not supported
        """
        if solver not in cls._solvers:
            available = ", ".join(cls.get_available_solvers())
            raise ValueError(f"Solver '{solver}' not supported. "
                          f"Available solvers: {available}")
        
        solver_class = cls._solvers[solver]
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
        solver = config.get("solver", "dwave")
        
        if solver not in cls._solvers:
            available = ", ".join(cls.get_available_solvers())
            raise ValueError(f"Solver '{solver}' not supported. "
                          f"Available solvers: {available}")
        
        solver_class = cls._solvers[solver]
        return solver_class.from_config(config)
    
    @classmethod
    def switch_solver(cls, solver: BaseSolver, new_solver: str, 
                      **kwargs) -> BaseSolver:
        """
        Switch a solver to a different solver while preserving common parameters.
        
        Args:
            solver: Current solver instance
            new_solver: New solver name
            **kwargs: Additional parameters for the new solver
            
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
        
        return cls.create_solver(new_solver, **common_params)


class DynamicSolver:
    """
    A wrapper class that allows dynamic solver switching during runtime.
    """
    
    def __init__(self, initial_solver: str = "dwave", **kwargs):
        """
        Initialize with a specific solver.
        
        Args:
            initial_solver: Initial solver to use
            **kwargs: Solver parameters
        """
        self.current_solver = initial_solver
        self.solver = SolverFactory.create_solver(initial_solver, **kwargs)
    
    def switch_solver(self, new_solver: str, **kwargs):
        """
        Switch to a different solver.
        
        Args:
            new_solver: New solver name
            **kwargs: Additional parameters for the new solver
        """
        self.solver = SolverFactory.switch_solver(
            self.solver, new_solver, **kwargs
        )
        self.current_solver = new_solver
    
    def get_solver_info(self) -> Dict[str, Any]:
        """
        Get information about the current solver.
        
        Returns:
            Dictionary with solver information
        """
        return self.solver.get_solver_info()
    
    def solve_qubo(self, builder):
        """
        Solve QUBO using the current solver.
        
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