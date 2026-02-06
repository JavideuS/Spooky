"""
Demo 3: Visualization

Purpose: Demonstrate all visualization capabilities
Difficulty: Beginner
Prerequisites: Understanding of demo 01_quickstart.py

This demo shows how to create beautiful visualizations of your quantum pathfinding
results, including static plots, step-by-step animations, and custom images.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import using package path
import quantum.visualizer as visualizer
from quantum.pathFormulation import PathfindingProblem
from quantum.builder import GraphQUBO
from quantum.solvers import SolverFactory
import quantum.config.parser as config_parser
from quantum.utils.logger import set_verbose_level

def main():
    print("=" * 70)
    print("Demo 3: Visualization - Creating Beautiful Plots")
    print("=" * 70)
    
    # Define base path
    base_path = Path(__file__).parent.parent
    
    # Step 1: Solve a simple problem
    print("\n[1/3] Solving a pathfinding problem...")
    
    config_path = base_path / "config/config.yaml"
    penalty_sets = config_parser.load_config(str(config_path), sections=["penalty_sets"])
    penalties_conf = penalty_sets["penalty_sets"]["crash"]
    
    # Set verbose level to 1 to only show the final result
    set_verbose_level(1)

    map_path = base_path / "maps/synthetic/3x3/obs3x3_standard"
    
    # Load materials for visualization colors
    materials_path = base_path / "config/materials.yaml"
    materials_data = config_parser.load_config(str(materials_path))["materials"]
    
    problem = PathfindingProblem.from_map_config(
        str(map_path),
        problem_name="baseline",
        materials_data=materials_data
    )
    
    print(f"    ✓ Map: {problem.grid.M}x{problem.grid.N}")
    print(f"    ✓ Obstacles: {problem.grid.obstacles}")
    
    p_graph = problem.as_graph_only()
    builder = GraphQUBO(p_graph, penalties=penalties_conf, name="viz_demo")
    
    solver = SolverFactory.create_solver(solver="dwave", normalize_scale=4, num_reads=10)

    solution = solver.solve_qubo_smart(builder, False)
    
    raw_path = solver.decode_path(solution["solution"], p_graph)
    # Extract path for the single robot (format: [(x, y, t), ...])
    # The visualizer expects coordinates, not the full ((x,y,t), robot_id) format
    robot_paths = solver.get_robot_paths(raw_path)
    path = list(robot_paths.values())[0] if robot_paths else []
    
    print(f"    ✓ Path: {path}")
    print(f"    ✓ Path found: {len(path)} steps")
    
    # Step 2: Create visualizer
    print("\n[2/3] Creating visualizer...")
    
    grid_size = (problem.grid.M, problem.grid.N)
    start = problem.robots["Lucia"].current_position
    goal = problem.robots["Lucia"].goal
    obstacles = problem.grid.obstacles
    
    # Option A: Basic visualizer (no custom images)
    viz = visualizer.QuantumRoboticsVisualizer(
        grid_size,
        title="Quantum Pathfinding - 3x3 Grid"
        
    )
    
    # Option B: Visualizer with custom images (uncomment to use)
    # images_dir = base_path / "images"
    # viz = visualizer.QuantumRoboticsVisualizer(
    #     grid_size,
    #     title="Quantum Pathfinding with Custom Images",
    #     start_image_path=str(images_dir / "scooby.svg"),
    #     goal_image_path=str(images_dir / "scoobysnack.svg"),
    #     obstacle_image_path=str(images_dir / "ghost.png")
    # )
    
    print("    ✓ Visualizer created")
    
    # Step 3: Create visualizations
    print("\n[3/3] Generating plots...")
    
    # --- Static Plot at Step 3 ---
    print("\n  Creating static plot (current position at step 3)...")
    static_fig = viz.create_static_plot(
        obstacles=obstacles,
        path=path,
        start=start,
        goal=goal,
        current_step=3,
        problem=problem
    )
    
    # Save to HTML (interactive)
    viz.write_html(static_fig, "visualization_static.html")
    viz.show(static_fig) # Popup for user
    print("    ✓ Saved to: visualization_static.html")
    
    # Save to PNG (static image - requires kaleido)
    try:
        viz.write_image(static_fig, "output/static_path.png")
        print("    ✓ Saved to: output/static_path.png")
    except Exception as e:
        print(f"    ⚠ Could not save PNG: {e}")
        print("      Install kaleido: pip install kaleido")
    
    # --- Step-by-Step Plot ---
    print("\n  Creating step-by-step plot (all timesteps)...")
    step_fig = viz.create_step_by_step_plot(
        obstacles=obstacles,
        path=path,
        start=start,
        goal=goal,
        problem=problem
    )
    
    # Save to HTML
    viz.write_html(step_fig, "output/step_by_step_path.html")
    print("    ✓ Saved to: output/step_by_step_path.html")
    viz.show(step_fig) # Popup for user
    
    # Save to PNG (larger size recommended for step-by-step)
    try:
        viz.write_image(step_fig, "output/step_by_step_path.png", width=1200, height=800)
        print("    ✓ Saved to: output/step_by_step_path.png")
    except Exception as e:
        print(f"    ⚠ Could not save PNG: {e}")
    
    # Display in notebook (if running in Jupyter)
    # viz.show(static_fig)
    # viz.show(step_fig)
    
    print("\n" + "=" * 70)
    print("✅ Demo complete!")
    print("\nVisualization files created:")
    print("  - output/static_path.html (open in browser)")
    print("  - output/step_by_step_path.html (open in browser)")
    print("\nVisualization features:")
    print("  ✓ Interactive hover to see coordinates")
    print("  ✓ Custom images for robots and obstacles")
    print("  ✓ Terrain background rendering")
    print("  ✓ Step-by-step path evolution")
    print("\nNext steps:")
    print("  - Open the HTML files in your browser to explore")
    print("  - Try demo 04_multi_robot.py for multi-agent visualization")
    print("  - Customize with your own images!")
    print("=" * 70)

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    import os
    os.makedirs("output", exist_ok=True)
    
    main()
