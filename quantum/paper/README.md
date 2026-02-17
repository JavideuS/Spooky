# Paper Visualization Tools

[![arXiv](https://img.shields.io/badge/arXiv-2602.14799-b31b1b.svg)](https://arxiv.org/abs/2602.14799)

**Paper:** [Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization](https://arxiv.org/abs/2602.14799)

**Citation:**

```bibtex
@misc{gonzalezvillasmil2026scalablemultirobotpathplanning,
      title={Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization},
      author={Javier González Villasmil et al.},
      year={2026},
      eprint={2602.14799},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.14799},
}
```

This directory contains tools for generating publication-quality plots comparing classical and quantum multi-robot path planning approaches.

## Files

- **`plot_comparison.py`** - Core visualization module with plotting functions
- **`generate_plots.py`** - Standalone CLI script for generating plots from benchmark results
- **`paper_plots.py`** - Legacy plotting utilities (QUBO heatmaps, energy landscapes, etc.)

## Quick Start

### Generate plots from a benchmark result:

````bash
# Basic usage - quantum paths only
python paper/generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json

# Compare with classical solver (Dijkstra)
python paper/generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json --compare-classical

## Classical Solver Algorithms

The visualization system includes proper multi-robot classical solvers for fair comparison:

### Single Robot
- **A*** (default): Optimal pathfinding with Manhattan distance heuristic
- **Dijkstra**: Guaranteed shortest path

### Multi-Robot (Prioritized Planning)
- Plans robots sequentially by priority (matching QUBO robot priorities)
- Uses **Space-Time A*** with collision avoidance
- Checks vertex collisions (same position at same time)
- Guarantees collision-free paths

**Usage:**
```bash
# Use prioritized planning for multi-robot (default)
python generate_plots.py --benchmark results/benchmarks/multi_robot.json --compare-classical

# Disable collision avoidance (independent paths)
python generate_plots.py --benchmark results/benchmarks/multi_robot.json --compare-classical --no-prioritized

# Use Dijkstra for single robot
python generate_plots.py --benchmark results/benchmarks/single_robot.json --compare-classical --algorithm dijkstra
````

# Use A\* instead of Dijkstra

python paper/generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json --compare-classical --algorithm astar

# Export in multiple formats

python paper/generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json --formats png pdf svg

# Specify custom output directory

python paper/generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json --output my_figures/

````

### Generate scalability analysis from multiple benchmarks:

```bash
# Analyze scalability across multiple benchmark files
python paper/generate_plots.py --scalability "results/benchmarks/benchmark_*.json" --output paper_figures/
````

## Using the API Directly

You can also use the visualization functions directly in your Python code:

```python
from paper.plot_comparison import (
    load_benchmark_results,
    extract_paths_from_benchmark,
    run_classical_solver,
    plot_path_comparison,
    plot_performance_metrics,
    plot_multi_robot_paths
)

# Load benchmark results
benchmark_data = load_benchmark_results('results/benchmarks/benchmark_20260201.json')

# Extract quantum paths
quantum_paths = extract_paths_from_benchmark(benchmark_data)

# Run classical solver for comparison
problem = ...  # Your PathfindingProblem instance
classical_paths = run_classical_solver(problem, algorithm='dijkstra')

# Generate comparison plot
plot_path_comparison(
    quantum_paths,
    classical_paths,
    problem,
    'figures/path_comparison.png'
)

# Generate performance metrics
classical_results = {
    'paths': classical_paths,
    'execution_time_sec': 0.123
}
plot_performance_metrics(
    benchmark_data,
    classical_results,
    'figures/performance.png'
)

# Visualize multi-robot paths
plot_multi_robot_paths(
    quantum_paths,
    problem,
    'figures/multi_robot.png'
)
```

## Available Plot Types

### 1. Path Comparison

Side-by-side visualization of classical vs quantum paths on the same grid.

**Function:** `plot_path_comparison(quantum_paths, classical_paths, problem, output_path)`

### 2. Performance Metrics

Bar charts comparing:

- Execution time
- Total path length
- Optimality gap

**Function:** `plot_performance_metrics(benchmark_results, classical_results, output_path)`

### 3. Multi-Robot Paths

Visualization of all robot paths on a single grid, with optional timestep filtering.

**Function:** `plot_multi_robot_paths(robot_paths, problem, output_path, timestep=None)`

### 4. Scalability Analysis

Dual-axis plot showing how execution time and qubit count scale with number of robots.

**Function:** `plot_scalability_analysis(results_list, output_path)`

## Integration with Benchmark System

The visualization tools are designed to work seamlessly with benchmark results:

```python
from benchmark.benchmark import BenchmarkRunner
from paper.plot_comparison import run_classical_solver, plot_path_comparison

# Run benchmark
runner = BenchmarkRunner(qubobuilder, solver, num_runs=10, level=2)
results = runner.run_build()

# Generate plots
quantum_paths = extract_paths_from_benchmark(results)
classical_paths = run_classical_solver(runner.problem)
plot_path_comparison(quantum_paths, classical_paths, runner.problem, 'comparison.png')
```

## Output Formats

Supported formats:

- **PNG** - Default, good for web and presentations
- **PDF** - Vector format, ideal for LaTeX papers
- **SVG** - Vector format, editable in Inkscape/Illustrator

All plots are generated at 300 DPI for publication quality.

## Classical Solver Options

Two classical algorithms are supported:

- **Dijkstra** - Optimal shortest path, no heuristic
- **A\*** - Optimal shortest path with Manhattan distance heuristic (faster)

Both algorithms use NetworkX and guarantee optimal solutions for comparison.

## Examples

See `examples/visualization_example.py` for a complete working example.

## Requirements

- matplotlib
- numpy
- networkx
- pathlib (standard library)

All dependencies are included in the main project requirements.
