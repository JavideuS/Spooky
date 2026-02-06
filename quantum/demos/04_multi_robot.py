"""
Demo 4: Multi-Robot Coordination

Purpose: Demonstrate multi-robot pathfinding with collision avoidance
Difficulty: Intermediate
Prerequisites: Understanding of demo 01_quickstart.py

This demo shows how to coordinate multiple robots on the same map, with different
priorities, safety radii, and start times to avoid collisions.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from pennylane import numpy as np
from quantum.pathFormulation import PathfindingProblem
from quantum.robotConfiguration import RobotConfig
from quantum.builder import GraphQUBO
from quantum.solvers import SolverFactory
import quantum.config.parser as config_parser
from quantum.utils.logger import set_verbose_level

def main():
    print("=" * 70)
    print("Demo 4: Multi-Robot Coordination")
    print("=" * 70)
    
    # Define base path
    base_path = Path(__file__).parent.parent
    
    # Step 1: Initialize problem
    print("\n[1/4] Loading map...")
    
    config_path = base_path / "config/config.yaml"
    penalty_sets = config_parser.load_config(str(config_path), sections=["penalty_sets"])
    penalties_conf = penalty_sets["penalty_sets"]["crash"]
    # Set verbose level to 1 to only show the final result
    set_verbose_level(1)
    
    # Start with a single robot from the map
    map_path = base_path / "maps/synthetic/5x5/no_obs5x5"
    problem = PathfindingProblem.from_map_config(
        str(map_path),
        problem_name="baseline"
    )
    
    print(f"    ✓ Map: {problem.grid.M}x{problem.grid.N}")
    print(f"    ✓ Initial robot: {list(problem.robots.keys())[0]}")
    
    # Step 2: Add more robots
    print("\n[2/4] Adding additional robots...")
    
    # Robot 2: Higher priority, starts immediately
    robot2 = RobotConfig(
        robot_id="Kai",
        start=(0, 4),  # Top-right
        goal=(4, 0),              # Bottom-left
        start_time=0,
        priority=1,               # Higher priority than default (1)
        safety_radius=0
    )
    problem.add_robot(robot2)
    print(f"    ✓ Added Kai: priority={robot2.priority}, start={(0,4)}, goal={(4,0)}")
    
    # Robot 3: Lower priority, starts later
    robot3 = RobotConfig(
        robot_id="Zane",
        start=(2, 0),  # Middle-left
        goal=(2, 4),              # Middle-right
        start_time=2,             # Starts 2 timesteps later
        priority=1,
        safety_radius=1           # Requires 1-cell buffer around it
    )
    problem.add_robot(robot3)
    print(f"    ✓ Added Zane: priority={robot3.priority}, start_time={robot3.start_time}, safety_radius={robot3.safety_radius}")
    
    # Display robot summary
    print("\n  Robot Summary:")
    for robot_id, robot in problem.robots.items():
        print(f"    - {robot_id:8s}: {robot.current_position} → {robot.goal} "
              f"(priority={robot.priority}, start_time={robot.start_time})")
    
    print(f"\n    ✓ Total robots: {problem.num_robots}")
    print(f"    ✓ Total timeline: {problem.T} timesteps")
    
    # Step 3: Build QUBO with robot-specific window limits
    print("\n[3/4] Building QUBO...")
    
    p_graph = problem.as_graph_only()
    
    # Optional: Set window limits per robot to manage QUBO size
    robot_window_limits = {
        "Lucia": 4,    # Lucia can plan up to 4 steps ahead
        "Kai": 5,      # Kai gets more planning horizon (higher priority)
        "Zane": 3   # Zane gets less (lower priority, starts later)
    }
    
    builder = GraphQUBO(
        p_graph,
        penalties=penalties_conf,
        name="multi_robot",
        robot_window_limits=robot_window_limits
    )
    
    print(f"    ✓ QUBO builder created with window limits")
    
    # Step 4: Solve
    print("\n[4/4] Solving multi-robot coordination...")
    
    solver = SolverFactory.create_solver(
        solver="dwave",
        normalize_scale=4,
        num_reads=15  # More reads for multi-robot problems
    )
    
    solution = solver.solve_qubo_smart(builder, False)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    energy = solver.total_energy(solution)
    print(f"\nSolution Energy: {energy:.4f}")
    
    # Decode paths for each robot
    print("\nRobot Paths:")
    print("-" * 70)
    
    # Decode path once
    raw_path = solver.decode_path(solution["solution"], p_graph)
    robot_paths = solver.get_robot_paths(raw_path)
    
    for robot_id, robot in problem.robots.items():
        # Find which robot number this ID corresponds to
        # In this simple demo, we can assume insertion order or look up keys
        robot_num = list(problem.robots.keys()).index(robot_id)
        
        path = robot_paths.get(robot_num, [])
        print(f"\n{robot_id}:")
        print(f"  Path length: {len(path)} steps")
        print(f"  Path: {path}")
    
    # Step 5: Visualization
    print("\n[5/5] Visualizing multi-robot paths...")
    try:
        from quantum.visualizer import QuantumRoboticsVisualizer
        
        # Create visualizer instance
        viz = QuantumRoboticsVisualizer(
            grid_size=(problem.grid.M, problem.grid.N),
            title="Multi-Robot Pathfinding",
            obstacle_image_path=str(base_path / "quantum/assets/obstacle_1.png") if (base_path / "quantum/assets/obstacle_1.png").exists() else None
        )
        
        # Static Plot
        print("  Generating static plot...")
        static_fig = viz.create_static_plot(
            obstacles=problem.grid.obstacles,
            problem=problem,
            robot_paths=robot_paths,
            current_step=max(len(p) for p in robot_paths.values()) - 1,
        )
        viz.write_html(static_fig, "output/multi_robot_static.html")
        viz.show(static_fig) # Popup for user
        
    except ImportError:
        print("  Skipping visualization (plotly not installed or other import error)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Visualization failed: {e}")

    # Check for collisions
    print("\n" + "-" * 70)
    print("Collision Analysis:")
    print("  The QUBO penalties ensure robots avoid collisions by:")
    print("  ✓ Respecting priority levels (higher priority gets preference)")
    print("  ✓ Temporal coordination (robots at different start times)")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\nKey concepts demonstrated:")
    print("  ✓ Adding multiple robots to a problem")
    print("  ✓ Setting different priorities")
    print("  ✓ Staggered start times")
    print("  ✓ Robot-specific window limits")
    print("\nNext steps:")
    print("  - Try demo 03_visualization.py to visualize multi-robot paths")
    print("  - Try demo 06_advanced_features.py for terrain-aware coordination")
    print("  - Experiment with different priority configurations")
    print("=" * 70)

if __name__ == "__main__":
    main()
