"""
Demo 6: Advanced Features - Automatic API

Purpose: Showcase easy robot addition, solver switching, and configuration experiments
Difficulty: Advanced
Prerequisites: Understanding of demo 01 (quickstart)

This demo shows how to quickly experiment with advanced features using the
automatic/convenient API - no manual boilerplate needed! Perfect for rapid
prototyping and testing different configurations.

Note: Demo 02 covers manual setup. This demo focuses on the fast, automatic approach.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pennylane import numpy as np
from quantum.pathFormulation import PathfindingProblem
from quantum.robotConfiguration import RobotConfig
from quantum.builder import GraphQUBO
from quantum.solvers import SolverFactory, DynamicSolver
import quantum.config.parser as config_parser
from quantum.utils.logger import set_verbose_level, get_logger

def main():
    print("=" * 70)
    print("Demo 6: Advanced Features - Automatic API")
    print("=" * 70)
    
    # Define base path
    base_path = Path(__file__).parent.parent

    # Set verbose level
    set_verbose_level(1)
    
    # Step 1: Quick problem setup with terrain
    print("\n[1/6] Quick setup with terrain map...")
    
    materials_path = base_path / "config/materials.yaml"
    materials_data = config_parser.load_config(str(materials_path))["materials"]
    
    # One-line initialization!
    map_path = base_path / "maps/synthetic/3x3/no_obs3x3_ter"
    problem = PathfindingProblem.from_map_config(
        str(map_path),
        problem_name="baseline",
        materials_data=materials_data
    )
    
    print(f"    ✓ Map loaded: {problem.grid.M}x{problem.grid.N}")
    print(f"    ✓ Initial robot: {list(problem.robots.keys())[0]}")
    
    # Step 2: Easily add more robots
    print("\n[2/6] Adding robots with one-liners...")
    
    # Just create and add - no manual configuration needed!
    robot2 = RobotConfig("Kai", (0, 2), (2, 0), start_time=0, priority=2)
    problem.add_robot(robot2)
    print(f"    ✓ Added Kai: {robot2.current_position} → {robot2.goal}")
    
    robot3 = RobotConfig("Jay", (1, 0), (1, 2), start_time=1, priority=1, safety_radius=1)
    problem.add_robot(robot3)
    print(f"    ✓ Added Jay: {robot3.current_position} → {robot3.goal}")
    
    print(f"\n    ✓ Total robots: {problem.num_robots}")
    print(f"    ✓ Timeline auto-calculated: {problem.T} timesteps")
    
    # Step 3: Try different penalty sets easily
    print("\n[3/6] Experimenting with penalty sets...")
    
    config_path = base_path / "config/config.yaml"
    penalty_sets = config_parser.load_config(str(config_path), sections=["penalty_sets"])
    
    # Compare different penalty configurations
    penalty_options = ["crash", "graph", "obs"]
    print(f"\n  Testing {len(penalty_options)} penalty sets:")
    
    for penalty_name in penalty_options:
        if penalty_name in penalty_sets["penalty_sets"]:
            penalties = penalty_sets["penalty_sets"][penalty_name]
            print(f"    - {penalty_name:8s}: K_hot={penalties['K_hot']}, K_adj={penalties['K_adj']}, K_goal={penalties['K_goal']}")
    
    # Use one for this demo
    penalties_conf = penalty_sets["penalty_sets"]["crash"]
    print(f"\n    ✓ Selected: 'crash' penalty set")
    
    # Step 4: Quick QUBO building
    print("\n[4/6] Building QUBO (automatic graph conversion)...")
    
    # Automatically convert to graph and build
    p_graph = problem.as_graph_only()
    builder = GraphQUBO(p_graph, penalties=penalties_conf, name="advanced_demo", robot_window_limits={"Lucia": 6, "Kai": 6, "Jay": 6})
    
    print(f"    ✓ QUBO built for {problem.num_robots} robots")
    
    # Step 5: Use DynamicSolver for runtime switching
    print("\n[5/6] Using DynamicSolver for runtime backend switching...")
    
    # Create DynamicSolver starting with DWave
    print("\n  Creating DynamicSolver...")
    solver = DynamicSolver(
        initial_solver="dwave",
        normalize_scale=4,
        num_reads=10
    )
    print(f"    ✓ DynamicSolver created with DWave backend")
    print(f"    ✓ Current solver: {solver.current_solver}")
    
    # Step 6: Solve with multiple backends dynamically
    print("\n[6/6] Solving with multiple backends...")
    
    results = {}
    
    # Solve with DWave
    print("\n  Solving with DWave...")
    dwave_solution = solver.solve_qubo_smart(builder, False)
    dwave_raw_path = solver.decode_path(dwave_solution["solution"], p_graph)
    dwave_robot_paths = solver.get_robot_paths(dwave_raw_path)
    dwave_energy = solver.total_energy(dwave_solution)
    
    # Calculate dimensions
    dwave_total_steps = sum(len(p) for p in dwave_robot_paths.values())
    
    results["DWave"] = {
        "energy": dwave_energy,
        "robot_paths": dwave_robot_paths,
        "total_steps": dwave_total_steps
    }
    print(f"    ✓ DWave: Energy={dwave_energy:.4f}, Total steps={dwave_total_steps}")
    
    # Switch to Pennylane dynamically!
    print("\n  Switching to Pennylane QAOA...")
    init_params = np.array([[1.70579, 0.70321062], [0.49879231, 0.49412656]], requires_grad=True)
    solver.switch_solver(
        "pennylane",
        normalize_scale=1.0,
        num_reads="auto",
        layers=2,
        optimizer="QNG",
        opt_steps=30,
        device="lightning.gpu",
        params=init_params
    )
    print(f"    ✓ Switched to: {solver.current_solver}")
    
    # Solve with Pennylane
    print("\n  Solving with Pennylane...")
    pennylane_solution = solver.solve_qubo_smart(builder)
    pennylane_raw_path = solver.decode_path(pennylane_solution["solution"], p_graph)
    pennylane_robot_paths = solver.get_robot_paths(pennylane_raw_path)
    pennylane_energy = solver.total_energy(pennylane_solution)
    
    pennylane_total_steps = sum(len(p) for p in pennylane_robot_paths.values())
    
    results["Pennylane"] = {
        "energy": pennylane_energy,
        "robot_paths": pennylane_robot_paths,
        "total_steps": pennylane_total_steps
    }
    print(f"    ✓ Pennylane: Energy={pennylane_energy:.4f}, Total steps={pennylane_total_steps}")
    
    # Results comparison
    print("\n" + "=" * 70)
    print("SOLVER COMPARISON RESULTS")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("DWave Quantum Annealer:")
    print("-" * 70)
    print(f"  Energy: {results['DWave']['energy']:.4f}")
    print(f"  Total steps: {results['DWave']['total_steps']}")
    for r_num, p in results['DWave']['robot_paths'].items():
        print(f"  - Robot {r_num}: {p}")
    
    print("\n" + "-" * 70)
    print("Pennylane QAOA (GPU):")
    print("-" * 70)
    print(f"  Energy: {results['Pennylane']['energy']:.4f}")
    print(f"  Total steps: {results['Pennylane']['total_steps']}")
    for r_num, p in results['Pennylane']['robot_paths'].items():
        print(f"  - Robot {r_num}: {p}")
    
    # Determine best solution
    print("\n" + "-" * 70)
    print("Comparison:")
    print("-" * 70)
    
    best_solver = "DWave" if results["DWave"]["energy"] < results["Pennylane"]["energy"] else "Pennylane"
    print(f"  Best energy: {best_solver} ({results[best_solver]['energy']:.4f})")
    
    if results["DWave"]["total_steps"] == results["Pennylane"]["total_steps"]:
        print(f"  Total steps: Both found {results['DWave']['total_steps']}-step solutions")
    else:
        shorter = "DWave" if results["DWave"]["total_steps"] < results["Pennylane"]["total_steps"] else "Pennylane"
        print(f"  Fewer steps: {shorter} ({results[shorter]['total_steps']} steps)")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\nKey features demonstrated:")
    print("  ✓ One-line problem initialization")
    print("  ✓ Easy robot addition (just create & add)")
    print("  ✓ Quick penalty set comparison")
    print("  ✓ Automatic QUBO building")
    print("  ✓ DynamicSolver for runtime backend switching!")
    print("  ✓ Solved same problem with 2 different backends")
    print("  ✓ Side-by-side result comparison")
    print("\nHow DynamicSolver works:")
    print("  1. Create with initial backend: DynamicSolver('dwave', ...)")
    print("  2. Solve: solver.solve_qubo(builder)")
    print("  3. Switch backend: solver.switch_solver('pennylane', ...)")
    print("  4. Solve again: solver.solve_qubo(builder)")
    print("  → Same solver object, different backends!")
    print("\nNext steps:")
    print("  - Combine with demo 03 to visualize both solutions")
    print("  - Use demo 02 when you need fine-grained control")
    print("  - Try demo 05 for real quantum hardware")
    print("=" * 70)

if __name__ == "__main__":
    main()
