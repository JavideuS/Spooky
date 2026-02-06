# Benchmarking

This module provides tools for evaluating the performance and reliability of the quantum solvers. It allows for running repeated trials of pathfinding problems to gather statistics on success rates, solution quality, and execution time.

## Components

### `BenchmarkRunner`
The core class that orchestrates the benchmarking process.

- **Multiple Runs**: Executes the solver a specified number of times (`num_runs`) on the same problem instance.
- **Validation**: Automatically validates each solution to ensure it meets all constraints (valid moves, no collisions, correct start/end).
- **Data Collection**: Aggregates results including:
  - Success/Failure status
  - Energy of the solution
  - Execution time
  - Decoded paths
- **Storage**: Saves detailed results to JSON files in `results/benchmarks`.
- **Verbosity Levels**: Control output detail and memory usage with 3 benchmark levels.

## Benchmark Levels

The `level` parameter controls the verbosity and memory footprint of benchmark results. This is particularly useful when running large-scale benchmarks (e.g., 1000+ runs) where memory consumption can become significant.

### Level 1: Summary Only (Minimal Memory)
**Use case**: Large-scale performance testing, CI/CD pipelines, quick validation

**Includes:**
- Run ID and timestamp
- Validation pass/fail status
- Total energy
- Execution time

**Excludes:**
- Robot paths
- Raw bit solutions
- Per-robot validation details

**Memory**: ~100-200 bytes per run

### Level 2: Paths Included (Default)
**Use case**: Standard benchmarking, path analysis, debugging multi-robot coordination

**Includes:**
- Everything from Level 1
- Robot paths for each run
- Per-robot validation details
- Per-window energies (for windowed solving)

**Excludes:**
- Raw bit solutions

**Memory**: ~500-2000 bytes per run (depends on path length and number of robots)

### Level 3: Full Debug (Maximum Detail)
**Use case**: Deep debugging, QUBO analysis, solution verification

**Includes:**
- Everything from Level 2
- Raw bit solution (complete QUBO variable assignment)

**Excludes:**
- Nothing (full detail)

**Memory**: ~2000-10000 bytes per run (depends on QUBO size)

## Usage

The benchmark runner is typically invoked within a script like `qubo.py`.

### Basic Usage (Default Level 2)

```python
from quantum.benchmark import BenchmarkRunner

# Initialize benchmark with a builder and a solver
runner = BenchmarkRunner(
    qubobuilder=my_builder,
    solver=my_solver,
    num_runs=100,
    output_dir="results/benchmarks"
)

# Execute the benchmark
runner.run_build()
```

### Memory-Optimized Benchmarking (Level 1)

```python
# For large-scale benchmarks (1000+ runs)
runner = BenchmarkRunner(
    qubobuilder=my_builder,
    solver=my_solver,
    num_runs=1000,
    level=1  # Summary only - minimal memory
)
runner.run_build()
```

### Full Debug Mode (Level 3)

```python
# For debugging QUBO formulations
runner = BenchmarkRunner(
    qubobuilder=my_builder,
    solver=my_solver,
    num_runs=10,
    level=3  # Include raw bit solutions
)
runner.run_build()
```

## Output Format

Benchmark results are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "problem": { ... },
    "solver": { ... },
    "penalty_set": { ... },
    "benchmark_level": 2,
    "num_runs": 100,
    "timestamp": "2026-01-26T20:00:00"
  },
  "runs": [
    {
      "run_id": 1,
      "timestamp": "2026-01-26T20:00:01",
      "valid": true,
      "energy": 123.45,
      "execution_time_sec": 2.5,
      
      // Level 2+ only:
      "robot_paths": {
        "Robot1": [[0, 0], [0, 1], ...],
        "Robot2": [[5, 5], [5, 4], ...]
      },
      "validation_details": { ... },
      "window_energies": [45.2, 38.1, 40.15],
      
      // Level 3 only:
      "raw_solution": [0, 1, 0, 1, 1, 0, ...]
    }
  ]
}
```

## Validation Logic

The benchmarking tool includes robust validation logic (`is_solution_valid`) that supports both Grid and Graph problem notations. It checks:
- **Continuity**: Robots must move between connected nodes/cells.
- **Obstacles**: Robots must not traverse blocked areas.
- **One-Hot**: Robots must be in exactly one place at one time.
- **Multi-Robot**: Robots must not collide with each other.
- **Start/Goal**: Robots must start and end at correct positions.
- **Early Stopping**: Supports robots reaching goals before time horizon.

## Performance Considerations

### Memory Usage Estimates

For a typical multi-robot problem with 3 robots and 20 timesteps:

| Level | Memory per Run | 100 Runs | 1000 Runs |
|-------|---------------|----------|-----------|
| 1     | ~150 bytes    | ~15 KB   | ~150 KB   |
| 2     | ~1.5 KB       | ~150 KB  | ~1.5 MB   |
| 3     | ~5 KB         | ~500 KB  | ~5 MB     |

**Recommendation**: Use Level 1 for runs > 500, Level 2 for standard benchmarks, Level 3 only for debugging specific issues.

