"""
Demo 2: Manual Setup

Purpose: Learn manual initialization for fine-grained control
Difficulty: Intermediate
Prerequisites: Understanding of demo 01_quickstart.py

This demo shows how to manually load maps, configure penalties, and initialize
problems when you need more control than the automatic initialization provides.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pennylane import numpy as np
from quantum.pathFormulation import PathfindingProblem
from quantum.builder import QUBOBuilder, GraphQUBO
from quantum.solvers import SolverFactory
import quantum.config.parser as config_parser
from quantum.config.hdf5parser import load_map_from_hdf5, load_both_from_hdf5
import quantum.map as map
from quantum.utils.logger import set_verbose_level

def main():
    print("=" * 70)
    print("Demo 2: Manual Setup - Step-by-Step Initialization")
    print("=" * 70)
    
    # Define base path relative to this script (quantum/demos/02_manual_setup.py -> quantum/)
    base_path = Path(__file__).parent.parent
    
    # Step 1: Manually load the HDF5 map file
    print("\n[1/7] Loading HDF5 map file...")
    map_path = base_path / "maps/synthetic/5x5/obs5x5_medium.h5"
    map_data = load_both_from_hdf5(str(map_path))
    
    print(f"    ✓ Map name: {map_data['name']}")
    print(f"    ✓ Has grid data: {map_data['has_map']}")
    print(f"    ✓ Has graph data: {map_data['has_graph']}")
    
    # Step 2: Load YAML configuration manually
    print("\n[2/7] Loading YAML configuration...")
    yaml_path = base_path / "maps/synthetic/5x5/obs5x5_medium.yaml"
    map_config = config_parser.load_config(str(yaml_path), sections=["problems"])
    problem_config = map_config["problems"]["baseline"]
    
    print(f"    ✓ Problem: baseline")
    print(f"    ✓ Start: {problem_config['start']}")
    print(f"    ✓ Goal: {problem_config['goal']}")
    print(f"    ✓ Time limit: {problem_config.get('time_limit', 'Auto')}")
    
    # Step 3: Load materials data (optional, for terrain costs)
    print("\n[3/7] Loading materials data...")
    materials_path = base_path / "config/materials.yaml"
    materials_data = config_parser.load_config(str(materials_path))["materials"]
    print(f"    ✓ Materials loaded: {len(materials_data)} types")
    
    # Step 4: Create Grid and Graph objects manually
    print("\n[4/7] Creating Grid and Graph representations...")
    
    # Create Grid
    grid = map.Grid.from_hdf5_data(
        map_data['map_data'],
        materials_data=materials_data,
        name=map_data['name']
    )
    print(f"    ✓ Grid: {grid.M}x{grid.N} with {len(grid.obstacles)} obstacles")
    
    # Create Graph
    graph = map.Graph.from_hdf5_data(
        map_data['graph_data'],
        name=map_data['name']
    )
    print(f"    ✓ Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Step 5: Initialize problem using from_unified_data
    print("\n[5/7] Initializing PathfindingProblem...")
    start = tuple(problem_config['start'])
    goal = tuple(problem_config['goal'])
    time_limit = problem_config.get('time_limit', None)
    
    problem = PathfindingProblem.from_unified_data(
        h5_source=map_path,
        start=start,
        end=goal,
        materials_data=materials_data,
        T=time_limit,
        name="manual_demo"
    )
    
    print(f"    ✓ Problem initialized: {problem.name}")
    print(f"    ✓ Timeline: {problem.T} timesteps")
    
    # Step 6: Configure penalties manually
    print("\n[6/7] Configuring QUBO penalties...")
    config_path = base_path / "config/config.yaml"
    penalty_sets = config_parser.load_config(str(config_path), sections=["penalty_sets"])
    
    # You can choose different penalty sets
    penalties_conf = penalty_sets["penalty_sets"]["crash"]
    
    print(f"    ✓ Penalty set: crash")
    print(f"    ✓ K_hot: {penalties_conf['K_hot']}")
    print(f"    ✓ K_adj: {penalties_conf['K_adj']}")
    print(f"    ✓ K_goal: {penalties_conf['K_goal']}")

    # Set verbose level to 1 to only show the final result
    set_verbose_level(1)
    
    # Step 7: Build QUBO (choose Grid or Graph representation)
    print("\n[7/7] Building QUBO...")
    
    # Option A: Use Grid representation
    # p_grid = problem.as_grid_only()
    # builder = QUBOBuilder(p_grid, penalties=penalties_conf, name="manual_grid")
    
    # Option B: Use Graph representation (more efficient)
    p_graph = problem.as_graph_only()
    builder = GraphQUBO(p_graph, penalties=penalties_conf, name="manual_graph")
    
    print(f"    ✓ QUBO builder created (Graph representation)")
    
    # Solve
    print("\n" + "=" * 70)
    print("SOLVING")
    print("=" * 70)
    
    solver = SolverFactory.create_solver(
        solver="dwave",
        normalize_scale=4,
        num_reads=5
    )
    
    builder.build()
    print("\nSolving with DWave quantum annealer...")
    solution = solver.solve_qubo_smart(builder, False)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Decode using the same problem options used for building (p_graph)
    # This ensures variable indices are interpreted correctly (Graph vs Grid)
    raw_path = solver.decode_path(solution["solution"], p_graph)
    # Extract path for the single robot
    robot_paths = solver.get_robot_paths(raw_path)
    path = list(robot_paths.values())[0] if robot_paths else []
    energy = solver.total_energy(solution)
    
    print(f"\n✓ Solution found!")
    print(f"  Energy: {energy:.4f}")
    print(f"  Path length: {len(path)} steps")
    print(f"\n  Path: {path}")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\nKey takeaways:")
    print("  - Manual setup gives you full control over initialization")
    print("  - You can choose between Grid and Graph representations")
    print("  - Different penalty sets affect solution quality")
    print("\nNext steps:")
    print("  - Try demo 03_visualization.py to visualize this path")
    print("  - Try demo 04_multi_robot.py for multi-agent scenarios")
    print("=" * 70)

if __name__ == "__main__":
    main()
