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
- **Extensible**: Easy to add new backends
- **Type Safety**: Full type hints for better IDE support

## Common Methods

All solvers provide these common methods:

- `solve_qubo(builder)`: Solve the QUBO problem
- `decode_path(sample, problem)`: Decode binary solution to path
- `total_energy(solution)`: Calculate total energy
- `to_dict()`: Get solver parameters as dictionary
- `get_backend_info()`: Get backend-specific information 
## Calibrated defaults: QAOA angles and shots, SA reads

The PennyLane solver runs QAOA with **fixed angles**: CLI, benchmark and sweep
runs never optimise them per window (training takes 10-20 minutes and gains
little run to run). With no `params`, 2-layer QAOA uses
`TUNED_ANGLES_2_LAYERS` and `num_reads="auto"` uses the `AUTO_SHOTS` table,
both in `Pennylane_solver.py`. Params are shaped `(2, layers)`: row 0 the
gammas, row 1 the betas.

Both come from `quantum/benchmark/calibration.py`, which records the
windows a problem family actually produces and evaluates QAOA on them exactly
(statevector, up to 20 qubits):

```bash
# pin heavy runs, e.g. taskset -c 0-11, to leave the machine usable
python -m quantum.benchmark.calibration record --config sweep_configs/synthetic_full.yaml
python -m quantum.benchmark.calibration angles --split size   # train small maps, test large
python -m quantum.benchmark.calibration shots --angles G1 G2 B1 B2
python -m quantum.benchmark.calibration record --config sweep_configs/synthetic_full.yaml --preprocess bfs_safe
python -m quantum.benchmark.calibration sa-reads --config sweep_configs/synthetic_full.yaml
```

Every command writes a JSON report to `results/calibration/` with the git
commit, arguments, machine and results, so a default in the code can be
traced to the run that produced it.

SA's `num_reads="auto"` (`DWaveSolver.AUTO_READS_PER_VAR`, reads =
max(4, ceil(c * vars)) per window) is calibrated end to end rather than on the
exact minimum: SA rarely reaches a large window's minimum (median ~600 reads
for a 99% hit at 61-100 variables) yet finds valid paths with far fewer. On the
synthetic family (bfs_safe, 37 problems x 5 runs) c = 1/8, 1/4, 1/2, 1 gave
97, 124, 152, 162 valid of 185 at 0.13-0.38 s per run; c = 1/2 is the smallest
within noise of the best, c = 1 the accuracy setting (gains mostly on 10x10).

Lessons from calibrating them:

- **Tune on a family, check on held-out problems.** Angles optimised on one
  instance (or a few) fit those Hamiltonians and can cancel the minimum on
  others. The previous hand-tuned angles concentrated some windows ~10x and
  sent the minimum's probability to ~0 on others -- on held-out windows they
  scored about the same as uniform sampling.
- **Score concentration, not hit rate.** "Did any shot hit the minimum"
  saturates: most windows are small enough that a few thousand shots cover
  the whole space whatever the circuit does, so unrelated angle sets tie at
  100%. `angles` maximises the mean log10 ratio of the minimum's probability
  to uniform sampling's, clipped so a blind spot costs a fixed amount.
- **Shots follow the angles.** The old shot table (500-25,000) was sized for
  brute-force coverage. With angles that concentrate, the worst window up to
  20 qubits needs ~400 shots for a 99% hit, so the table is ~10x smaller.
- **Check out of family, not just held-out problems.** A held-out split from
  the same family can't see distribution shift (other maps, robot counts,
  penalties). `--split size` trains on maps of at most 25 cells and tests on
  larger ones: the angles came out the same (gammas [1.045, 2.611], betas
  [-0.441, -0.238], test +1.17), so they track the formulation, not the sample.
- **Recalibrate** whenever the goal term, penalties, preprocessing or angles
  change -- angles transfer across similar Hamiltonians (the same optimum
  came out of the legacy and shaped goal terms), but that has to be checked,
  not assumed. Above 20 qubits and on hardware (noise weakens concentration)
  the shot table is an extrapolation.
