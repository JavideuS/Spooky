"""
Quantum Solvers Package

This package provides a flexible architecture for quantum solvers with support
for multiple backends including DWave, Pennylane, and others.

Main classes:
- BaseSolver: Abstract base class for all solvers
- SolverFactory: Factory for creating and managing solvers
- DynamicSolver: Wrapper for runtime backend switching
- DWaveSolver: DWave quantum annealing solver
- PennylaneSolver: Pennylane QAOA solver
- ILPSolver: Exact ILP solver via Pyomo (classical baseline)
- CBSSolver: Conflict-Based Search solver (classical baseline)
"""

from .base_solver import BaseSolver
from .solver_factory import SolverFactory, DynamicSolver
from .DWave_solver import DWaveSolver
from .Pennylane_solver import PennylaneSolver
from .ILP_solver import ILPSolver
from .CBS_solver import CBSSolver

__all__ = [
    "BaseSolver",
    "SolverFactory",
    "DynamicSolver",
    "DWaveSolver",
    "PennylaneSolver",
    "ILPSolver",
    "CBSSolver",
]
