"""
Paper visualization utilities for quantum vs classical multi-robot path planning.

This module provides tools for generating publication-quality comparison plots.

Main modules:
- plot_comparison: Core visualization functions
- generate_plots: CLI script for generating plots from benchmarks

Quick start:
    # From command line
    python paper/generate_plots.py --benchmark results/benchmarks/benchmark_*.json --compare-classical
    
    # From Python
    from paper.plot_comparison import plot_path_comparison, run_classical_solver
"""

from .plot_comparison import (
    load_benchmark_results,
    extract_paths_from_benchmark,
    run_classical_solver,
    compare_solutions,
    plot_path_comparison,
    plot_performance_metrics,
    plot_multi_robot_paths,
    plot_scalability_analysis,
    aggregate_multiple_runs
)

__all__ = [
    'load_benchmark_results',
    'extract_paths_from_benchmark',
    'run_classical_solver',
    'compare_solutions',
    'plot_path_comparison',
    'plot_performance_metrics',
    'plot_multi_robot_paths',
    'plot_scalability_analysis',
    'aggregate_multiple_runs'
]
