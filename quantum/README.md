# Quantum Navigation Core

The `quantum` package is the heart of the Quantum Navigation project. It implements a hybrid quantum-classical approach to multi-robot path planning, utilizing QUBO (Quadratic Unconstrained Binary Optimization) formulations to solve complex navigation tasks.

For theoretical background and formulation details, please refer to our paper: [**Scalable Multi-Robot Path Planning via Quadratic Unconstrained Binary Optimization**](https://arxiv.org/abs/2602.14799).

## Package Structure

- **`builder/`**: Constructs the mathematical models (QUBOs) from problem definitions. Contains logic for grid and graph-based environments.
- **`solvers/`**: Interfaces for various quantum and classical solvers (DWave, PennyLane, Simulated Annealing).
- **`config/`**: Configuration management, file parsing (YAML, HDF5), and ROS integration tools.
- **`utils/`**: Shared utility functions for path handling and data manipulation.
- **`benchmark/`**: Tools for benchmarking solver performance and accuracy.

## Core Modules

### Problem Formulation

- **`pathFormulation.py`**: Defines the mathematical formulation of the pathfinding problem. It translates high-level constraints (start, goal, obstacles) into the logic needed by the builders.
- **`map.py`**: Handles the internal representation of the environment, including grid data, obstacles, terrain costs, and graph topology.
- **`robotConfiguration.py`**: Manages the state and parameters of individual robots within the swarm.

### Visualization

- **`visualizer.py`**: Tools for visualizing the navigation process, including:
  - 2D grid maps with paths.
  - Energy landscapes of the quantum solution.
  - Real-time solving progress.

## Getting Started

To run a navigation task or benchmark, you can use the command-line interface `qubo_cli.py` for flexibility, or the `qubo.py` script for hands-on code modification.

### Using the CLI (`qubo_cli.py`)

The CLI allows you to specify maps, problems, solvers, and run benchmarks directly from the terminal:

```bash
# DWave example
python qubo_cli.py --map maps/synthetic/10x10/obs10x10_hard --problem four_robots

# Pennylane example
python qubo_cli.py --map maps/synthetic/10x10/no_obs10x10 --problem two_robots --var-limit 605 --solver pennylane --benchmark --num-runs 1
```

Run `python qubo_cli.py --help` to see all available options.

### Using the Script (`qubo.py`)

1. **Configure**: Edit `config/config.yaml` to set your map and solver preferences.
2. **Run**: Execute the main script.

```bash
python qubo.py
```

**Tip**: Control console output verbosity by setting `verbose.level` in `config/config.yaml` (or via `--verbose` in the CLI):

- `0` = Silent (errors only)
- `1` = Minimal (essential info)
- `2` = Standard (default)
- `3` = Debug (all details)

Inside `qubo.py`, you can switch between different problem configurations (e.g., grid vs. graph) and solvers by uncommenting the relevant lines.

## Key Concepts

- **Windowing**: To overcome the qubit limitations of current quantum hardware, paths are solved in "sliding windows" (e.g., 5 steps at a time) rather than all at once.
- **Hybrid Solving**: The system can dynamically switch between quantum annealers (DWave), gate-based QAOA (PennyLane), and classical heuristics depending on problem complexity and resource availability.
