"""
Visualization utilities for comparing classical and quantum multi-robot path planning.

This module provides functions to generate publication-quality plots from benchmark
results, including comparisons between classical (Dijkstra/A*) and quantum (QUBO) approaches.
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_solver_name(solver_info) -> str:
    """
    Format solver information for display in plots.
    
    Args:
        solver_info: Solver name string or dict with solver config
        
    Returns:
        Formatted string like "QUBO (D-Wave Simulated)" or "QUBO (QAOA)"
    """
    if isinstance(solver_info, dict):
        solver_type = solver_info.get('solver', 'unknown')
    else:
        solver_type = str(solver_info)
    
    # Map solver types to display names
    solver_map = {
        'dwave': 'QUBO (D-Wave Simulated)',
        'pennylane': 'QUBO (QAOA)',
        'pennylane_qaoa': 'QUBO (QAOA)',
        'qiskit': 'QUBO (Qiskit)',
        'qiskit.remote': 'QUBO (Qiskit Hardware)',
        'qiskit.aer': 'QUBO (Qiskit Aer)',
        'pennylane_qiskit_remote': 'QUBO (Qiskit Hardware)',
    }
    
    return solver_map.get(solver_type.lower(), f'QUBO ({solver_type})')



def load_benchmark_results(json_path: Union[str, Path]) -> Dict:
    """
    Load benchmark results from JSON file.
    
    Args:
        json_path: Path to benchmark JSON file
        
    Returns:
        Dictionary containing benchmark results
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_paths_from_benchmark(benchmark_data: Dict) -> Dict[str, List]:
    """
    Extract robot paths from benchmark results.
    
    Args:
        benchmark_data: Benchmark results dictionary
        
    Returns:
        Dictionary mapping robot IDs to their paths
    """
    if 'runs' not in benchmark_data or not benchmark_data['runs']:
        return {}
    
    # Get first valid run (or could aggregate across runs)
    for run in benchmark_data['runs']:
        if run.get('valid', False) and 'robot_paths' in run:
            return run['robot_paths']
    
    return {}


def run_classical_solver(
    problem,
    algorithm: str = 'astar',
    use_prioritized: bool = True,
    verbose: bool = False
) -> Dict[str, List]:
    """
    Run classical pathfinding algorithm on the problem.
    
    For single robot: Uses standard A* or Dijkstra
    For multi-robot: Uses prioritized planning with space-time A* and collision avoidance
    
    Args:
        problem: PathfindingProblem instance
        algorithm: 'dijkstra' or 'astar' (for single robot)
        use_prioritized: Use prioritized planning for multi-robot (default: True)
        verbose: Print detailed solving information (default: False)
        
    Returns:
        Dictionary mapping robot IDs to classical paths
    """
    num_robots = len(problem.robots)
    
    # Multi-robot case: use prioritized planning with collision avoidance
    if num_robots > 1 and use_prioritized:
        try:
            from paper.classical_mapf import run_prioritized_planning
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"PRIORITIZED PLANNING WITH COLLISION AVOIDANCE")
                print(f"{'='*60}")
                print(f"Number of robots: {num_robots}")
                print(f"Planning order (by priority):")
                sorted_robots = sorted(
                    problem.robots.items(),
                    key=lambda x: getattr(x[1], 'priority', 1.0),
                    reverse=True
                )
                for i, (robot_id, robot) in enumerate(sorted_robots, 1):
                    priority = getattr(robot, 'priority', 1.0)
                    print(f"  {i}. {robot_id} (priority={priority:.1f}): {robot.start} → {robot.goal}")
                print()
            else:
                print(f"  Using prioritized planning with collision avoidance ({num_robots} robots)")
            
            import time
            start = time.time()
            paths = run_prioritized_planning(problem, algorithm='astar', verbose=verbose)
            elapsed = time.time() - start
            
            if verbose:
                print(f"\nPlanning completed in {elapsed:.4f}s")
                print(f"\nResults:")
                for robot_id, path in paths.items():
                    if path:
                        print(f"  {robot_id}: {len(path)} steps")
                    else:
                        print(f"  {robot_id}: NO PATH FOUND")
                print(f"{'='*60}\n")
            
            return paths
            
        except Exception as e:
            print(f"  Warning: Prioritized planning failed: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            print(f"  Falling back to independent path planning")
            # Fall through to independent planning
    
    # Single robot or fallback: independent path planning
    classical_paths = {}
    
    # Build NetworkX graph from problem
    G = nx.Graph()
    
    if hasattr(problem, 'graph') and problem.graph is not None:
        # Build from graph
        graph = problem.graph
        
        # Add nodes with positions
        for idx, pos in enumerate(graph.nodes):
            G.add_node(idx, pos=tuple(pos))
        
        # Add edges
        for edge in graph.edges:
            if len(edge) == 2:
                i, j = edge
                G.add_edge(int(i), int(j), weight=1.0)
            else:
                i, j, w = edge
                G.add_edge(int(i), int(j), weight=float(w))
                
    elif hasattr(problem, 'grid') and problem.grid is not None:
        # Build from grid
        grid = problem.grid
        
        # Add nodes for non-obstacle cells
        for r in range(grid.M):
            for c in range(grid.N):
                if (r, c) not in grid.obstacles:
                    node_id = r * grid.N + c
                    G.add_node(node_id, pos=(r, c))
        
        # Add edges from adjacency
        for (r, c), neighbors in grid.adjacency.items():
            # Skip if source is obstacle
            if (r, c) in grid.obstacles:
                continue
                
            u = r * grid.N + c
            for (nr, nc) in neighbors:
                # Skip if neighbor is obstacle
                if (nr, nc) in grid.obstacles:
                    continue
                    
                v = nr * grid.N + nc
                if u < v:  # Avoid duplicates
                    G.add_edge(u, v, weight=1.0)
    else:
        raise ValueError("Problem must have either graph or grid")
    
    # Solve for each robot independently
    if num_robots == 1:
        print(f"  Using standard {algorithm.upper()} (single robot)")
    else:
        print(f"  Using independent {algorithm.upper()} (no collision avoidance)")
    
    for robot_id, robot in problem.robots.items():
        # Convert start/goal to node IDs if needed
        if hasattr(problem, 'graph') and problem.graph is not None:
            # For graph problems, start/goal might be node IDs or positions
            if isinstance(robot.start, tuple):
                start_node = problem.graph.get_node_from_position(robot.start)
                goal_node = problem.graph.get_node_from_position(robot.goal)
            else:
                start_node = robot.start
                goal_node = robot.goal
        else:
            # For grid problems, convert (r,c) to node ID
            if isinstance(robot.start, tuple):
                start_node = robot.start[0] * problem.grid.N + robot.start[1]
                goal_node = robot.goal[0] * problem.grid.N + robot.goal[1]
            else:
                start_node = robot.start
                goal_node = robot.goal
        
        # Run algorithm
        try:
            if algorithm == 'astar':
                # A* with Manhattan distance heuristic
                def heuristic(a, b):
                    pos_a = G.nodes[a].get('pos', (0, 0))
                    pos_b = G.nodes[b].get('pos', (0, 0))
                    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
                
                path_nodes = nx.astar_path(G, start_node, goal_node, heuristic=heuristic, weight='weight')
            else:
                path_nodes = nx.dijkstra_path(G, start_node, goal_node, weight='weight')
            
            # Convert to coordinate format with timesteps
            path = []
            for t, node in enumerate(path_nodes):
                pos = G.nodes[node].get('pos')
                if pos:
                    # Add timestep (start from robot's start_time if available)
                    start_time = getattr(robot, 'start_time', 0)
                    path.append((*pos, t + start_time))
                else:
                    # Fallback if no position stored
                    path.append((node, 0, t))
            
            classical_paths[robot_id] = path
            
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"Warning: No path found for robot {robot_id}: {e}")
            classical_paths[robot_id] = []
    
    return classical_paths


def compare_solutions(quantum_paths: Dict, classical_paths: Dict) -> Dict[str, Any]:
    """
    Compute comparison metrics between quantum and classical solutions.
    
    Args:
        quantum_paths: Dictionary of robot paths from quantum solver
        classical_paths: Dictionary of robot paths from classical solver
        
    Returns:
        Dictionary of comparison metrics
    """
    metrics = {
        'path_length_comparison': {},
        'optimality_gap': {},
        'total_quantum_length': 0,
        'total_classical_length': 0,
    }
    
    for robot_id in quantum_paths.keys():
        q_path = quantum_paths.get(robot_id, [])
        c_path = classical_paths.get(robot_id, [])
        
        q_len = len(q_path)
        c_len = len(c_path)
        
        metrics['path_length_comparison'][robot_id] = {
            'quantum': q_len,
            'classical': c_len,
            'difference': q_len - c_len
        }
        
        if c_len > 0:
            metrics['optimality_gap'][robot_id] = (q_len - c_len) / c_len * 100
        else:
            metrics['optimality_gap'][robot_id] = 0
        
        metrics['total_quantum_length'] += q_len
        metrics['total_classical_length'] += c_len
    
    # Overall optimality gap
    if metrics['total_classical_length'] > 0:
        metrics['overall_optimality_gap'] = (
            (metrics['total_quantum_length'] - metrics['total_classical_length']) 
            / metrics['total_classical_length'] * 100
        )
    else:
        metrics['overall_optimality_gap'] = 0
    
    return metrics


def plot_path_comparison(
    quantum_paths: Dict[str, List],
    classical_paths: Dict[str, List],
    problem,
    output_path: str,
    solver_name: str = "QUBO",
    classical_algorithm: str = "A*",
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 300
):
    """
    Create side-by-side visualization of classical vs quantum paths.
    
    Args:
        quantum_paths: Dictionary mapping robot IDs to quantum paths
        classical_paths: Dictionary mapping robot IDs to classical paths
        problem: PathfindingProblem instance
        output_path: Where to save the plot
        figsize: Figure size in inches
        dpi: Resolution for saved image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get grid dimensions
    if hasattr(problem, 'grid') and problem.grid is not None:
        M, N = problem.grid.M, problem.grid.N
        obstacles = problem.grid.obstacles
    else:
        # Estimate from graph
        M = N = int(np.sqrt(len(problem.graph.nodes))) + 1
        obstacles = set()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(quantum_paths)))
    
    for ax, paths, title in [(ax1, classical_paths, f'Classical ({classical_algorithm})'),
                              (ax2, quantum_paths, f'{solver_name}')]:
        # Setup grid
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(-0.5, M - 0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(N))
        ax.set_yticks(range(M))
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
        ax.invert_yaxis()
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw obstacles
        for obs in obstacles:
            rect = patches.Rectangle(
                (obs[1] - 0.5, obs[0] - 0.5), 1, 1,
                linewidth=0, facecolor='gray', alpha=0.7
            )
            ax.add_patch(rect)
        
        # Draw paths for each robot
        for idx, (robot_id, path) in enumerate(paths.items()):
            if not path:
                continue
            
            color = colors[idx]
            robot = problem.robots[robot_id]
            
            # Extract positions (remove timestep)
            positions = [(p[0], p[1]) if len(p) >= 2 else p for p in path]
            
            if len(positions) > 1:
                # Draw path line
                rows, cols = zip(*positions)
                ax.plot(cols, rows, color=color, linewidth=2, alpha=0.7, 
                       label=f'{robot_id} (len={len(path)})')
                
                # Draw path points
                ax.scatter(cols, rows, color=color, s=30, alpha=0.5, zorder=3)
            
            # Mark start
            start_pos = positions[0]
            ax.scatter(start_pos[1], start_pos[0], color=color, s=200, 
                      marker='o', edgecolors='black', linewidths=2, zorder=4)
            
            # Mark goal
            goal_pos = positions[-1] if positions else robot.goal
            if isinstance(goal_pos, tuple) and len(goal_pos) >= 2:
                ax.scatter(goal_pos[1], goal_pos[0], color=color, s=200,
                          marker='*', edgecolors='black', linewidths=2, zorder=4)
        
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved path comparison to {output_path}")


def plot_performance_metrics(
    benchmark_results: Dict,
    classical_results: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 300
):
    """
    Create bar charts comparing performance metrics.
    
    Args:
        benchmark_results: Quantum benchmark results
        classical_results: Classical solver results (with timing, path lengths)
        output_path: Where to save the plot
        figsize: Figure size
        dpi: Resolution
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Extract metrics
    # Exclude first run (has initialization overhead) and average the rest
    run_times = [run['execution_time_sec'] for run in benchmark_results['runs']]
    if len(run_times) > 1:
        quantum_time = np.mean(run_times[1:])  # Exclude first run
    else:
        quantum_time = run_times[0] if run_times else 0
    
    classical_time = classical_results.get('execution_time_sec', 0)
    
    quantum_energy = np.mean([run['energy'] for run in benchmark_results['runs']])
    
    # Path lengths
    quantum_paths = extract_paths_from_benchmark(benchmark_results)
    classical_paths = classical_results.get('paths', {})
    
    quantum_total_len = sum(len(p) for p in quantum_paths.values())
    classical_total_len = sum(len(p) for p in classical_paths.values())
    
    # Extract solver name from benchmark metadata
    solver_name = benchmark_results.get('metadata', {}).get('solver', 'QUBO')
    
    # Format solver name for display
    solver_display = format_solver_name(solver_name)
    
    # Plot 1: Execution Time
    ax = axes[0]
    methods = ['Classical', solver_display]
    times = [classical_time, quantum_time]
    bars = ax.bar(methods, times, color=['#2ecc71', '#3498db'], alpha=0.8)
    ax.set_ylabel('Execution Time (s)', fontsize=11)
    ax.set_title('Solver Time Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Path Length
    ax = axes[1]
    lengths = [classical_total_len, quantum_total_len]
    bars = ax.bar(methods, lengths, color=['#2ecc71', '#3498db'], alpha=0.8)
    ax.set_ylabel('Total Path Length (steps)', fontsize=11)
    ax.set_title('Path Length Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{length}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Optimality Gap
    ax = axes[2]
    if classical_total_len > 0:
        gap = ((quantum_total_len - classical_total_len) / classical_total_len) * 100
    else:
        gap = 0
    
    color = '#e74c3c' if gap > 0 else '#2ecc71'
    bar = ax.bar(['Optimality Gap'], [gap], color=color, alpha=0.8)
    ax.set_ylabel('Gap (%)', fontsize=11)
    ax.set_title('Solution Quality', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    ax.text(0, gap, f'{gap:.1f}%', ha='center', 
           va='bottom' if gap > 0 else 'top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved performance metrics to {output_path}")


def plot_multi_robot_paths(
    robot_paths: Dict[str, List],
    problem,
    output_path: str,
    timestep: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 300
):
    """
    Visualize multi-robot paths on grid, optionally at specific timestep.
    
    Args:
        robot_paths: Dictionary mapping robot IDs to paths
        problem: PathfindingProblem instance
        output_path: Where to save the plot
        timestep: If specified, show robot positions at this timestep
        figsize: Figure size
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get grid dimensions
    if hasattr(problem, 'grid') and problem.grid is not None:
        M, N = problem.grid.M, problem.grid.N
        obstacles = problem.grid.obstacles
    else:
        M = N = int(np.sqrt(len(problem.graph.nodes))) + 1
        obstacles = set()
    
    # Setup grid
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, M - 0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(N))
    ax.set_yticks(range(M))
    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    if timestep is not None:
        ax.set_title(f'Multi-Robot Paths (t={timestep})', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Multi-Robot Paths (Complete)', fontsize=14, fontweight='bold')
    
    # Draw obstacles
    for obs in obstacles:
        rect = patches.Rectangle(
            (obs[1] - 0.5, obs[0] - 0.5), 1, 1,
            linewidth=0, facecolor='gray', alpha=0.7, label='Obstacle' if obs == list(obstacles)[0] else ''
        )
        ax.add_patch(rect)
    
    # Colors for robots
    colors = plt.cm.tab10(np.linspace(0, 1, len(robot_paths)))
    
    # Draw paths
    for idx, (robot_id, path) in enumerate(robot_paths.items()):
        if not path:
            continue
        
        color = colors[idx]
        robot = problem.robots[robot_id]
        
        # Extract positions
        positions = [(p[0], p[1]) if len(p) >= 2 else p for p in path]
        
        if timestep is not None:
            # Show path up to timestep and current position
            relevant_path = [p for p in path if len(p) >= 3 and p[2] <= timestep]
            if relevant_path:
                positions_to_t = [(p[0], p[1]) for p in relevant_path]
                
                if len(positions_to_t) > 1:
                    rows, cols = zip(*positions_to_t)
                    ax.plot(cols, rows, color=color, linewidth=2, alpha=0.5, linestyle='--')
                
                # Current position at timestep
                current_pos = positions_to_t[-1]
                ax.scatter(current_pos[1], current_pos[0], color=color, s=300,
                          marker='o', edgecolors='black', linewidths=2, 
                          label=f'{robot_id}', zorder=5)
                ax.text(current_pos[1], current_pos[0], robot_id, 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        else:
            # Show complete path
            if len(positions) > 1:
                rows, cols = zip(*positions)
                ax.plot(cols, rows, color=color, linewidth=2.5, alpha=0.7,
                       label=f'{robot_id} (len={len(path)})')
                ax.scatter(cols, rows, color=color, s=40, alpha=0.6, zorder=3)
            
            # Mark start
            start_pos = positions[0]
            ax.scatter(start_pos[1], start_pos[0], color=color, s=250,
                      marker='o', edgecolors='black', linewidths=2.5, zorder=4)
            ax.text(start_pos[1], start_pos[0], 'S', ha='center', va='center',
                   fontsize=9, fontweight='bold', color='white')
            
            # Mark goal
            goal_pos = positions[-1]
            ax.scatter(goal_pos[1], goal_pos[0], color=color, s=250,
                      marker='*', edgecolors='black', linewidths=2.5, zorder=4)
    
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-robot paths to {output_path}")


def plot_scalability_analysis(
    results_list: List[Dict],
    output_path: str,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 300
):
    """
    Plot scalability metrics (robots vs time/qubits).
    
    Args:
        results_list: List of benchmark results with varying number of robots
        output_path: Where to save the plot
        figsize: Figure size
        dpi: Resolution
    """
    # Extract data
    num_robots = []
    avg_times = []
    num_qubits = []
    
    for result in results_list:
        metadata = result.get('metadata', {})
        problem_data = metadata.get('problem', {})
        
        n_robots = len(problem_data.get('robots', {}))
        num_robots.append(n_robots)
        
        # Average execution time
        runs = result.get('runs', [])
        if runs:
            avg_time = np.mean([run['execution_time_sec'] for run in runs])
            avg_times.append(avg_time)
        else:
            avg_times.append(0)
        
        # Estimate qubits (would need to be stored in metadata)
        # For now, approximate based on problem size
        T = problem_data.get('T', 10)
        grid_size = problem_data.get('grid', {}).get('M', 5) * problem_data.get('grid', {}).get('N', 5)
        qubits = n_robots * T * grid_size
        num_qubits.append(qubits)
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=figsize)
    
    color1 = '#e74c3c'
    ax1.set_xlabel('Number of Robots', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (s)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(num_robots, avg_times, marker='o', color=color1, linewidth=2.5,
                     markersize=8, label='Execution Time')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    
    # Second y-axis for qubits
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.set_ylabel('Number of Qubits', color=color2, fontsize=12, fontweight='bold')
    line2 = ax2.plot(num_robots, num_qubits, marker='s', color=color2, linewidth=2.5,
                     markersize=8, linestyle='--', label='Qubits')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title and legend
    ax1.set_title('Scalability Analysis: Multi-Robot QUBO', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved scalability analysis to {output_path}")


def aggregate_multiple_runs(benchmark_files: List[Union[str, Path]]) -> List[Dict]:
    """
    Load and aggregate multiple benchmark result files.
    
    Args:
        benchmark_files: List of paths to benchmark JSON files
        
    Returns:
        List of benchmark result dictionaries
    """
    results = []
    for filepath in benchmark_files:
        try:
            data = load_benchmark_results(filepath)
            results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def plot_qubo_heatmap(problem, penalties: Dict, output_path: str):
    """
    Generate a heatmap visualization of the QUBO matrix.
    
    Args:
        problem: PathfindingProblem instance
        penalties: Dictionary of penalty weights
        output_path: Path to save the plot
    """
    try:
        from builder.QUBOBuilder import QUBOBuilder
        from builder.GraphQUBO import GraphQUBO
    except ImportError as e:
        print(f"Warning: Could not import QUBO builders: {e}")
        print("QUBO heatmap generation requires the builder module")
        return
    
    print(f"Building QUBO for heatmap visualization...")
    
    try:
        # Determine which builder to use based on problem type
        if hasattr(problem, 'grid') and problem.grid is not None:
            # Use QUBOBuilder for grid problems
            builder = QUBOBuilder(
                problem=problem,
                penalties=penalties,
                window_max_steps=min(problem.T, 10),  # Limit window size for visualization
                name="visualization_qubo"
            )
        elif hasattr(problem, 'graph') and problem.graph is not None:
            # Use GraphQUBO for graph problems
            builder = GraphQUBO(
                problem=problem,
                penalties=penalties,
                window_max_steps=min(problem.T, 10),
                name="visualization_qubo"
            )
        else:
            print("Warning: Problem has neither grid nor graph")
            return
        
        # Build with all constraints
        constraints = ["one_hot", "start", "goal", "adjacency", "obstacle", "crash"]
        builder.build(constraints_to_apply=constraints)
        
        Q = builder.Q
        
        if not Q:
            print("Warning: Empty QUBO matrix")
            return
        
        # Extract indices
        indices = set()
        for (i, j) in Q.keys():
            indices.add(i)
            indices.add(j)
        
        max_idx = max(indices)
        size = max_idx + 1
        
        # Create matrix
        matrix = np.zeros((size, size))
        for (i, j), val in Q.items():
            matrix[i, j] = val
            if i != j:
                matrix[j, i] = val  # Symmetric
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use seismic colormap for positive/negative values
        im = ax.imshow(matrix, cmap='seismic', interpolation='nearest', aspect='auto')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Coupling Strength')
        
        # Title and labels
        problem_type = "Grid" if hasattr(problem, 'grid') and problem.grid else "Graph"
        ax.set_title(f'QUBO Connectivity Heatmap ({problem_type})\n{len(problem.robots)} robots, T={problem.T}', 
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Variable Index', fontsize=12)
        ax.set_ylabel('Variable Index', fontsize=12)
        
        # Calculate sparsity
        sparsity = 1.0 - (np.count_nonzero(matrix) / matrix.size)
        num_vars = size
        num_couplings = np.count_nonzero(matrix)
        
        # Add statistics box
        stats_text = f'Variables: {num_vars}\nCouplings: {num_couplings}\nSparsity: {sparsity:.2%}'
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                fontsize=10,
                family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved QUBO heatmap to {output_path}")
        print(f"  Variables: {num_vars}, Couplings: {num_couplings}, Sparsity: {sparsity:.2%}")
        
    except Exception as e:
        print(f"Warning: Could not generate QUBO heatmap: {e}")
        import traceback
        traceback.print_exc()


def plot_circuit_diagram(problem, penalties: Dict, output_path: str, num_layers: int = 2):
    """
    Generate a circuit diagram for the QAOA ansatz.
    
    Note: This requires PennyLane and may not work for very large problems.
    
    Args:
        problem: PathfindingProblem instance
        penalties: Dictionary of penalty weights
        output_path: Path to save the plot
        num_layers: Number of QAOA layers to visualize
    """
    try:
        import pennylane as qml
        from builder.QUBOBuilder import QUBOBuilder
        from builder.GraphQUBO import GraphQUBO
        
        print(f"Building circuit diagram...")
        
        # Determine which builder to use
        if hasattr(problem, 'grid') and problem.grid is not None:
            builder = QUBOBuilder(
                problem=problem,
                penalties=penalties,
                window_max_steps=min(problem.T, 5),  # Keep small for circuit visualization
                name="circuit_viz"
            )
        elif hasattr(problem, 'graph') and problem.graph is not None:
            builder = GraphQUBO(
                problem=problem,
                penalties=penalties,
                window_max_steps=min(problem.T, 5),
                name="circuit_viz"
            )
        else:
            print("Warning: Problem has neither grid nor graph")
            return
        
        constraints = ["one_hot", "start", "goal", "adjacency"]
        builder.build(constraints_to_apply=constraints)
        
        # Get Ising Hamiltonian
        Hc, constant = builder.qubo_to_ising()
        wires = list(builder.get_wires())
        
        if not wires or len(wires) > 20:
            print(f"Warning: Circuit has {len(wires)} qubits, too large for clear visualization")
            if len(wires) > 20:
                print("Skipping circuit diagram (>20 qubits)")
                return
        
        num_qubits = len(wires)
        dev = qml.device("default.qubit", wires=wires)
        
        @qml.qnode(dev)
        def circuit(params):
            # Initial state (equal superposition)
            for w in wires:
                qml.Hadamard(wires=w)
            
            # QAOA layers
            for layer in range(num_layers):
                gamma = params[layer]
                beta = params[num_layers + layer]
                
                # Cost layer (apply Hc)
                for (i, j), coeff in Hc.items():
                    if i == j:
                        qml.RZ(2 * gamma * coeff, wires=i)
                    else:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * coeff, wires=j)
                        qml.CNOT(wires=[i, j])
                
                # Mixer layer
                for w in wires:
                    qml.RX(2 * beta, wires=w)
            
            return qml.probs(wires=wires)
        
        # Draw circuit with dummy parameters
        dummy_params = [0.5] * (2 * num_layers)
        fig, ax = qml.draw_mpl(circuit, style='pennylane')(dummy_params)
        
        # Add title
        problem_type = "Grid" if hasattr(problem, 'grid') and problem.grid else "Graph"
        fig.suptitle(f'QAOA Circuit ({problem_type}, {num_layers} layers, {num_qubits} qubits)', 
                     fontsize=14, fontweight='bold')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved circuit diagram to {output_path}")
        
    except ImportError:
        print("Warning: PennyLane not available, skipping circuit diagram")
    except Exception as e:
        print(f"Warning: Could not generate circuit diagram: {e}")
