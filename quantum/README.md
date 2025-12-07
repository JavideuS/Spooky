# Quantum Navigation Core

The `quantum` package is the heart of the Quantum Navigation project. It implements a hybrid quantum-classical approach to multi-robot path planning, utilizing QUBO (Quadratic Unconstrained Binary Optimization) formulations to solve complex navigation tasks.

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

### Execution & Orchestration
- **`multi_robot_solver.py`**: The main orchestrator that manages the solving loop. It handles:
  - Problem windowing (splitting long paths into solvable chunks).
  - Multi-robot coordination.
  - Calling the appropriate builder and solver backends.
  - Aggregating results.

### Visualization
- **`visualizer.py`**: Tools for visualizing the navigation process, including:
  - 2D grid maps with paths.
  - Energy landscapes of the quantum solution.
  - Real-time solving progress.

## Getting Started

To run a navigation task or benchmark, the primary entry point is `qubo.py`. You can modify this script to change scenarios, maps, and solver configurations.

1. **Configure**: Edit `config/config.yaml` to set your map and solver preferences.
2. **Run**: Execute the main script.

```bash
python qubo.py
```

Inside `qubo.py`, you can switch between different problem configurations (e.g., grid vs. graph) and solvers by uncommenting the relevant lines.

## Key Concepts

- **Windowing**: To overcome the qubit limitations of current quantum hardware, paths are solved in "sliding windows" (e.g., 10 steps at a time) rather than all at once.
- **Hybrid Solving**: The system can dynamically switch between quantum annealers (DWave), gate-based QAOA (PennyLane), and classical heuristics depending on problem complexity and resource availability.
