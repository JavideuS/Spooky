# Dynamic Quantum Solver Architecture

This package provides a flexible architecture for quantum solvers that supports multiple backends and dynamic switching between them.

## Architecture Overview

### Core Components

1. **BaseSolver**: Abstract base class that defines the common interface for all quantum solvers
2. **SolverFactory**: Factory class for creating and managing different solver backends
3. **DynamicSolver**: Wrapper class that allows runtime switching between backends
4. **Backend-Specific Solvers**: Concrete implementations for different quantum backends

### Supported Backends

- **DWave**: Quantum annealing using DWave's solvers
- **Pennylane**: QAOA (Quantum Approximate Optimization Algorithm) using Pennylane

## Usage Examples

### Method 1: Direct Solver Creation

```python
from solvers import SolverFactory

# Create a DWave solver
dwave_solver = SolverFactory.create_solver(
    backend="dwave", 
    normalize_scale=2.0, 
    num_reads=10
)

# Create a Pennylane solver
pennylane_solver = SolverFactory.create_solver(
    backend="pennylane", 
    normalize_scale=2.0, 
    num_reads=5, 
    layers=2
)
```

### Method 2: Configuration-Based Creation

```python
from solvers import SolverFactory

# Load configuration
config = {
    "backend": "dwave",
    "normalization_scale": 2.0,
    "num_reads": 10
}

solver = SolverFactory.create_solver_from_config(config)
```

### Method 3: Dynamic Backend Switching

```python
from solvers import DynamicSolver

# Start with DWave backend
dynamic_solver = DynamicSolver(
    backend="dwave", 
    normalize_scale=2.0, 
    num_reads=10
)

# Solve with DWave
solution_dwave = dynamic_solver.solve_qubo(qubo_builder)

# Switch to Pennylane backend
dynamic_solver.switch_backend("pennylane", layers=2, optimizer="adam")

# Solve with Pennylane
solution_pennylane = dynamic_solver.solve_qubo(qubo_builder)
```

### Method 4: Backward Compatibility

The old `QUBOSolver` class is still available for backward compatibility:

```python
from solvers import QUBOSolver

solver = QUBOSolver(normalize_scale=2.0, num_reads=10)
```

## Adding New Backends

To add a new quantum backend:

1. Create a new solver class that inherits from `BaseSolver`
2. Implement the abstract `solve_qubo` method
3. Register the solver with the factory

```python
from solvers import BaseSolver, SolverFactory

class CustomSolver(BaseSolver):
    def __init__(self, **kwargs):
        super().__init__(backend="custom", **kwargs)
    
    def solve_qubo(self, builder):
        # Your custom implementation here
        pass

# Register the new solver
SolverFactory.register_solver("custom", CustomSolver)
```

## Configuration

Solver configurations can be defined in the config file:

```yaml
solver:
  dwave_qa:
    backend: "dwave"
    normalization_scale: 1.0
    num_reads: 10

  pennylane_qaoa:
    backend: "pennylane"
    normalization_scale: 1.0
    num_reads: 10
    layers: 2
    optimizer: "adam"
```

## Key Features

- **Unified Interface**: All solvers implement the same interface
- **Dynamic Switching**: Switch between backends at runtime
- **Configuration Support**: Create solvers from configuration files
- **Backward Compatibility**: Old code continues to work
- **Extensible**: Easy to add new backends
- **Type Safety**: Full type hints for better IDE support

## Common Methods

All solvers provide these common methods:

- `solve_qubo(builder)`: Solve the QUBO problem
- `decode_path(sample, problem)`: Decode binary solution to path
- `total_energy(solution)`: Calculate total energy
- `to_dict()`: Get solver parameters as dictionary
- `get_backend_info()`: Get backend-specific information 