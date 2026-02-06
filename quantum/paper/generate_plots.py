#!/usr/bin/env python3
"""
Generate publication-quality plots for quantum vs classical multi-robot path planning.

This script loads benchmark results and generates comparison plots between
quantum (QUBO) and classical (Dijkstra/A*) approaches.

Usage:
    python generate_plots.py --benchmark results/benchmarks/benchmark_*.json --output paper_figures/
    python generate_plots.py --benchmark results/benchmarks/benchmark_*.json --compare-classical --algorithm astar
"""

import argparse
import sys
import time
from pathlib import Path
from glob import glob
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from paper.plot_comparison import (
    load_benchmark_results,
    extract_paths_from_benchmark,
    run_classical_solver,
    compare_solutions,
    plot_path_comparison,
    plot_performance_metrics,
    plot_multi_robot_paths,
    plot_scalability_analysis,
    aggregate_multiple_runs,
    plot_qubo_heatmap,
    plot_circuit_diagram
)
from pathFormulation import PathfindingProblem


def reconstruct_problem_from_metadata(metadata: dict) -> PathfindingProblem:
    """
    Reconstruct a PathfindingProblem from benchmark metadata.
    
    Args:
        metadata: Benchmark metadata dictionary
        
    Returns:
        PathfindingProblem instance
    """
    from map import Grid, Graph
    from robotConfiguration import RobotConfig
    
    problem_data = metadata.get('problem', {})
    
    # Reconstruct robots
    robots = []
    for robot_id, robot_data in problem_data.get('robots', {}).items():
        # Convert string keys back to tuples if needed
        start = robot_data.get('start')
        goal = robot_data.get('goal')
        
        # Handle tuple conversion from JSON strings or lists
        if isinstance(start, str) and start.startswith('('):
            start = eval(start)
        elif isinstance(start, list):
            start = tuple(start)
            
        if isinstance(goal, str) and goal.startswith('('):
            goal = eval(goal)
        elif isinstance(goal, list):
            goal = tuple(goal)
        
        # RobotConfig doesn't have T parameter, it has expected_duration
        # Also, start_time might not be in robot_data, default to 0
        robot = RobotConfig(
            robot_id=robot_id,
            start=start,
            goal=goal,
            priority=robot_data.get('priority', 1.0),
            start_time=0,  # Default to 0, not stored in benchmark metadata
            safety_radius=robot_data.get('safety_radius', 0.5),
            expected_duration=None  # Let problem calculate it
        )
        robots.append(robot)
    
    # Reconstruct grid or graph
    grid = None
    graph = None
    
    if 'grid' in problem_data:
        grid_data = problem_data['grid']
        obstacles = grid_data.get('obstacles', [])
        
        # Convert obstacle strings to tuples if needed
        obstacles_converted = []
        for obs in obstacles:
            if isinstance(obs, str) and obs.startswith('('):
                obstacles_converted.append(eval(obs))
            elif isinstance(obs, list):
                obstacles_converted.append(tuple(obs))
            else:
                obstacles_converted.append(obs)
        
        grid = Grid(
            M=grid_data.get('M', 5),
            N=grid_data.get('N', 5),
            obstacles=obstacles_converted,
            name=grid_data.get('name', 'grid')
        )
    
    if 'graph' in problem_data:
        graph_data = problem_data['graph']
        
        # Graph data is stored as numpy array strings, need to parse them
        try:
            import re
            
            # Parse nodes - format: "[[0 0]\n [0 1]\n ...]"
            nodes_str = graph_data.get('nodes', '')
            if nodes_str:
                # Remove brackets and split by newlines
                nodes_str = nodes_str.strip().replace('[', '').replace(']', '')
                node_lines = [line.strip() for line in nodes_str.split('\n') if line.strip()]
                nodes = []
                for line in node_lines:
                    coords = [int(x) for x in line.split()]
                    if len(coords) == 2:
                        nodes.append(coords)
            else:
                nodes = []
            
            # Parse edges - format: "[(0, 1, 1.0) (1, 2, 1.0) ...]"
            edges_str = graph_data.get('edges', '')
            if edges_str:
                # Extract tuples using regex
                edge_pattern = r'\(\s*(\d+),\s*(\d+),\s*([\d.]+)\s*\)'
                matches = re.findall(edge_pattern, edges_str)
                edges = [(int(i), int(j), float(w)) for i, j, w in matches]
            else:
                edges = []
            
            if nodes and edges:
                graph = Graph(
                    nodes=nodes,
                    edges=edges,
                    name=graph_data.get('name', 'graph')
                )
        except Exception as e:
            print(f"Warning: Could not parse graph data: {e}")
            graph = None
    
    # Create problem
    problem = PathfindingProblem(
        robots=robots,
        grid=grid,
        graph=graph,
        T=problem_data.get('T', 10),
        name=problem_data.get('name', 'problem')
    )
    
    return problem


def generate_all_plots(
    benchmark_path: str,
    output_dir: str,
    compare_classical: bool = True,
    classical_algorithm: str = 'astar',
    use_prioritized: bool = True,
    formats: list = ['png'],
    include_qubo: bool = True,
    include_circuit: bool = False
):
    """
    Generate all plots from a benchmark result file.
    
    Args:
        benchmark_path: Path to benchmark JSON file
        output_dir: Directory to save plots
        compare_classical: Whether to run classical solver for comparison
        classical_algorithm: 'dijkstra' or 'astar' (for single robot)
        use_prioritized: Use prioritized planning for multi-robot (default: True)
        formats: List of output formats (png, pdf, svg)
        include_qubo: Whether to generate QUBO heatmap
        include_circuit: Whether to generate circuit diagram (requires PennyLane)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating plots from: {benchmark_path}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # Load benchmark results
    print("Loading benchmark results...")
    benchmark_data = load_benchmark_results(benchmark_path)
    
    # Extract quantum paths
    quantum_paths = extract_paths_from_benchmark(benchmark_data)
    
    if not quantum_paths:
        print("Warning: No valid paths found in benchmark results")
        return
    
    # Reconstruct problem
    print("Reconstructing problem from metadata...")
    try:
        problem = reconstruct_problem_from_metadata(benchmark_data['metadata'])
        penalties = benchmark_data['metadata'].get('penalty_set', {})
    except Exception as e:
        print(f"Error reconstructing problem: {e}")
        print("Skipping plots that require problem instance")
        return
    
    # Run classical solver if requested
    classical_paths = {}
    classical_results = {}
    
    if compare_classical:
        print(f"\nRunning classical solver...")
        start_time = time.time()
        try:
            classical_paths = run_classical_solver(
                problem,
                algorithm=classical_algorithm,
                use_prioritized=use_prioritized
            )
            classical_time = time.time() - start_time
            
            # Print classical paths for debugging
            print(f"\nClassical paths found:")
            for robot_id, path in classical_paths.items():
                if path:
                    print(f"  {robot_id}: {len(path)} steps")
                    print(f"    Start: {path[0][:2]}, Goal: {path[-1][:2]}")
                    # Check for obstacle collisions
                    if hasattr(problem, 'grid') and problem.grid:
                        obstacles = problem.grid.obstacles
                        for i, (x, y, t) in enumerate(path):
                            if (x, y) in obstacles:
                                print(f"    ⚠️  WARNING: Step {i} at ({x},{y}) is an OBSTACLE!")
                    # Print full path if short enough
                    if len(path) <= 30:
                        print(f"    Path: {[p[:2] for p in path]}")
                else:
                    print(f"  {robot_id}: NO PATH FOUND")
            
            classical_results = {
                'paths': classical_paths,
                'execution_time_sec': classical_time
            }
            
            print(f"Classical solver completed in {classical_time:.3f}s")
            
            # Compute comparison metrics
            metrics = compare_solutions(quantum_paths, classical_paths)
            print(f"\nComparison Metrics:")
            print(f"  Quantum total path length: {metrics['total_quantum_length']}")
            print(f"  Classical total path length: {metrics['total_classical_length']}")
            print(f"  Overall optimality gap: {metrics['overall_optimality_gap']:.2f}%")
            
        except Exception as e:
            print(f"Warning: Classical solver failed: {e}")
            compare_classical = False
    
    # Generate plots
    base_name = Path(benchmark_path).stem
    
    # Extract solver info for plot labels
    solver_info = benchmark_data.get('metadata', {}).get('solver', 'QUBO')
    from plot_comparison import format_solver_name
    solver_display = format_solver_name(solver_info)
    
    # Determine classical algorithm name
    classical_algo_display = classical_algorithm.upper() if classical_algorithm else 'A*'
    
    print("\nGenerating plots...")
    
    # 1. Multi-robot paths (quantum only)
    for fmt in formats:
        output_file = output_path / f"{base_name}_quantum_paths.{fmt}"
        print(f"  - Multi-robot paths: {output_file}")
        plot_multi_robot_paths(quantum_paths, problem, str(output_file))
    
    # 2. Path comparison (if classical available)
    if compare_classical and classical_paths:
        for fmt in formats:
            output_file = output_path / f"{base_name}_path_comparison.{fmt}"
            print(f"  - Path comparison: {output_file}")
            plot_path_comparison(
                quantum_paths, 
                classical_paths, 
                problem, 
                str(output_file),
                solver_name=solver_display,
                classical_algorithm=classical_algo_display
            )
    
    # 3. Performance metrics (if classical available)
    if compare_classical and classical_results:
        for fmt in formats:
            output_file = output_path / f"{base_name}_performance.{fmt}"
            print(f"  - Performance metrics: {output_file}")
            plot_performance_metrics(benchmark_data, classical_results, str(output_file))
    
    # 4. QUBO heatmap
    if include_qubo and penalties:
        for fmt in formats:
            output_file = output_path / f"{base_name}_qubo_heatmap.{fmt}"
            print(f"  - QUBO heatmap: {output_file}")
            try:
                plot_qubo_heatmap(problem, penalties, str(output_file))
            except Exception as e:
                print(f"    Warning: Could not generate QUBO heatmap: {e}")
    
    # 5. Circuit diagram (optional, requires PennyLane)
    if include_circuit and penalties:
        for fmt in formats:
            output_file = output_path / f"{base_name}_circuit.{fmt}"
            print(f"  - Circuit diagram: {output_file}")
            try:
                plot_circuit_diagram(problem, penalties, str(output_file))
            except Exception as e:
                print(f"    Warning: Could not generate circuit diagram: {e}")
    
    print(f"\n✓ All plots generated successfully!")


def generate_scalability_plots(
    benchmark_pattern: str,
    output_dir: str,
    formats: list = ['png']
):
    """
    Generate scalability plots from multiple benchmark files.
    
    Args:
        benchmark_pattern: Glob pattern for benchmark files (e.g., "results/benchmarks/*.json")
        output_dir: Directory to save plots
        formats: List of output formats
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating scalability plots from: {benchmark_pattern}")
    print(f"{'='*60}\n")
    
    # Find all matching files
    benchmark_files = sorted(glob(benchmark_pattern))
    
    if not benchmark_files:
        print(f"No benchmark files found matching: {benchmark_pattern}")
        return
    
    print(f"Found {len(benchmark_files)} benchmark files")
    
    # Load all results
    results_list = aggregate_multiple_runs(benchmark_files)
    
    if not results_list:
        print("No valid results loaded")
        return
    
    # Generate scalability plot
    for fmt in formats:
        output_file = output_path / f"scalability_analysis.{fmt}"
        print(f"  - Scalability analysis: {output_file}")
        plot_scalability_analysis(results_list, str(output_file))
    
    print(f"\n✓ Scalability plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots for quantum vs classical path planning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots from a single benchmark
  python generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json
  
  # Compare with classical solver
  python generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json --compare-classical
  
  # Use A* instead of Dijkstra
  python generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json --compare-classical --algorithm astar
  
  # Generate scalability plots from multiple benchmarks
  python generate_plots.py --scalability "results/benchmarks/benchmark_*.json" --output paper_figures/
  
  # Export in multiple formats
  python generate_plots.py --benchmark results/benchmarks/benchmark_20260201.json --formats png pdf svg
        """
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        type=str,
        help='Path to benchmark JSON file'
    )
    
    parser.add_argument(
        '--scalability', '-s',
        type=str,
        help='Glob pattern for multiple benchmark files (for scalability analysis)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='paper_figures',
        help='Output directory for plots (default: paper_figures)'
    )
    
    parser.add_argument(
        '--compare-classical', '-c',
        action='store_true',
        help='Run classical solver for comparison'
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=['dijkstra', 'astar'],
        default='astar',
        help='Classical algorithm for single robot (default: astar)'
    )
    
    parser.add_argument(
        '--no-prioritized',
        action='store_false',
        dest='use_prioritized',
        help='Disable prioritized planning for multi-robot (use independent paths instead)'
    )
    
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=['png', 'pdf', 'svg'],
        default=['png'],
        help='Output formats (default: png)'
    )
    
    parser.add_argument(
        '--qubo',
        action='store_true',
        default=True,
        help='Generate QUBO heatmap (default: True)'
    )
    
    parser.add_argument(
        '--no-qubo',
        action='store_false',
        dest='qubo',
        help='Skip QUBO heatmap generation'
    )
    
    parser.add_argument(
        '--circuit',
        action='store_true',
        help='Generate circuit diagram (requires PennyLane, may be slow for large problems)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.benchmark and not args.scalability:
        parser.error("Either --benchmark or --scalability must be specified")
    
    # Generate plots
    try:
        if args.benchmark:
            generate_all_plots(
                args.benchmark,
                args.output,
                compare_classical=args.compare_classical,
                classical_algorithm=args.algorithm,
                use_prioritized=args.use_prioritized,
                formats=args.formats,
                include_qubo=args.qubo,
                include_circuit=args.circuit
            )
        
        if args.scalability:
            generate_scalability_plots(
                args.scalability,
                args.output,
                formats=args.formats
            )
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
