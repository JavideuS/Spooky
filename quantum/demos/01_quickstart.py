"""
Demo 1: Quickstart

Purpose: Get started in 30 seconds with automatic initialization
Difficulty: Beginner
Prerequisites: None - this is your starting point!

This demo shows the fastest way to set up and solve a quantum pathfinding problem.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pennylane import numpy as np
from quantum.pathFormulation import PathfindingProblem
from quantum.builder import GraphQUBO
from quantum.solvers import SolverFactory
import quantum.config.parser as config_parser
from quantum.utils.logger import set_verbose_level

def main():
    print("=" * 70)
    print("Demo 1: Quickstart - Automatic Initialization")
    print("=" * 70)
    
    # Define base path relative to this script (quantum/demos/01_quickstart.py -> quantum/)
    base_path = Path(__file__).parent.parent
    config_path = base_path / "config" / "config.yaml"
    
    # Step 1: Load penalties configuration
    print("\n[1/4] Loading QUBO penalties...")
    config = config_parser.load_config(str(config_path), sections=["penalty_sets", "verbose"])
    penalties_conf = config["penalty_sets"]["crash"]
    verbose_level = config["verbose"]["level"]
    set_verbose_level(verbose_level)
    
    # Step 2: Initialize problem using automatic map config loading
    print("[2/4] Loading problem from map configuration...")
    map_path = base_path / "maps/synthetic/5x5/no_obs5x5"
    problem = PathfindingProblem.from_map_config(
        str(map_path),
        problem_name="baseline"
    )
    
    print(f"    ✓ Map loaded: {problem.name}")
    print(f"    ✓ Grid size: {problem.grid.M}x{problem.grid.N}")
    print(f"    ✓ Robot: {list(problem.robots.keys())[0]}")
    print(f"    ✓ Start: {problem.robots['Lucia'].current_position}")
    print(f"    ✓ Goal: {problem.robots['Lucia'].goal}")
    
    # Step 3: Build QUBO
    print("\n[3/4] Building QUBO...")
    p_graph = problem.as_graph_only()
    builder = GraphQUBO(p_graph, penalties=penalties_conf, name="quickstart")
    # To visualize early stop and higher performance one usually limit to 5 steps per window, but in easy scenarios is not necessary
    # builder = GraphQUBO(p_graph, penalties=penalties_conf, name="quickstart", robot_window_limits={"Lucia": 5})
    
    print(f"    ✓ QUBO builder created")
    
    # Step 4: Solve with DWave
    print("\n[4/4] Solving with DWave quantum annealer...")
    solver = SolverFactory.create_solver(
        solver="dwave", 
        normalize_scale=4, 
        num_reads=5
    )
    
    builder.build()
    solution = solver.solve_qubo_smart(builder, False)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    path = solver.decode_path(solution["solution"], p_graph)
    energy = solver.total_energy(solution)
    
    print(f"\n✓ Solution found!")
    print(f"  Energy: {energy:.4f}")
    print(f"  Path length: {len(path)} steps")
    print(f"\n  Path: {path}")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\nNext steps:")
    print("  - Try demo 02_manual_setup.py to learn manual initialization")
    print("  - Try demo 03_visualization.py to visualize this path")
    print("=" * 70)

if __name__ == "__main__":
    main()
