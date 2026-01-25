"""
This module provides common validation functions used across QUBO builders,
solvers, and benchmarking tools.
"""

from typing import Tuple, Optional


def is_valid_move(
    problem,
    from_pos: Tuple[int, int],
    to_pos: Tuple[int, int],
    goal: Optional[Tuple[int, int]] = None
) -> bool:
    """
    Check if a move from one position to another is valid.
    
    Supports both grid and graph problem formats. Validates that positions
    are adjacent according to the problem's adjacency structure.
    
    Args:
        problem: Problem instance (grid or graph)
        from_pos: Starting position (i, j) coordinates
        to_pos: Destination position (i, j) coordinates
        goal: Optional goal position for goal-lock checking
        
    Returns:
        True if the move is valid (adjacent or same position), False otherwise
        
    Examples:
        >>> is_valid_move(problem, (0, 0), (0, 1))  # Adjacent cells
        True
        >>> is_valid_move(problem, (0, 0), (0, 0))  # Staying in place
        True
        >>> is_valid_move(problem, (0, 0), (5, 5))  # Non-adjacent
        False
        >>> is_valid_move(problem, (2, 3), (2, 3), goal=(2, 3))  # At goal
        True
    """
    # Same position is always valid (waiting/staying in place)
    if from_pos == to_pos:
        return True
    
    # Allow staying at goal position
    if goal is not None and from_pos == goal and to_pos == goal:
        return True
    
    problem_type = problem.get_format_type()
    
    if problem_type == "grid":
        # Check grid adjacency using adjacency dictionary
        return to_pos in problem.grid.adjacency.get(from_pos, [])
    else:  # graph or both
        # Check graph adjacency
        from_node = problem.graph.get_node_from_position(from_pos)
        to_node = problem.graph.get_node_from_position(to_pos)
        
        if from_node is None or to_node is None:
            return False
        
        # Adjacency list stores (neighbor_node, weight) tuples
        neighbors = [n for n, _ in problem.graph.adjacency.get(from_node, [])]
        return to_node in neighbors


def get_position_representation(problem, position: Tuple[int, int]):
    """
    Get the position representation appropriate for the problem type.
    
    For grid problems, returns the position coordinates as-is.
    For graph problems, returns the node ID corresponding to the position.
    
    Args:
        problem: Problem instance (grid or graph)
        position: Position tuple (i, j)
        
    Returns:
        For grid: position tuple (i, j)
        For graph: node ID (int)
        
    Examples:
        >>> get_position_representation(grid_problem, (2, 3))
        (2, 3)
        >>> get_position_representation(graph_problem, (2, 3))
        15  # Node ID at position (2, 3)
    """
    problem_type = problem.get_format_type()
    
    if problem_type == "grid":
        # Grid uses coordinates directly
        return position
    else:  # graph or both
        # Graph uses node IDs
        return problem.graph.get_node_from_position(position)
