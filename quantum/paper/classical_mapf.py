"""
Classical Multi-Agent Path Finding (MAPF) algorithms.

This module provides classical baseline algorithms for multi-robot path planning,
including space-time A* and prioritized planning with collision avoidance.
"""

import heapq
from typing import Dict, List, Tuple, Set, Optional, Union
import networkx as nx
from dataclasses import dataclass, field


@dataclass(order=True)
class Node:
    """Node in space-time A* search."""
    f_score: float
    g_score: float = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    time: int = field(compare=False)
    parent: Optional['Node'] = field(default=None, compare=False)


class SpaceTimeAStar:
    """
    A* pathfinding in space-time with collision avoidance.
    
    This algorithm finds paths while avoiding collisions with other robots
    by treating time as an additional dimension. It maintains a reservation
    table of occupied (position, time) tuples.
    """
    
    def __init__(self, graph: nx.Graph, max_time: int = 100):
        """
        Initialize Space-Time A*.
        
        Args:
            graph: NetworkX graph representing the environment
            max_time: Maximum time horizon for planning
        """
        self.graph = graph
        self.max_time = max_time
        self.reservations: Set[Tuple[Tuple[int, int], int]] = set()
    
    def add_reservation(self, position: Tuple[int, int], time: int):
        """Reserve a position at a specific time."""
        self.reservations.add((position, time))
    
    def add_path_reservations(self, path: List[Tuple[int, int, int]]):
        """
        Reserve all positions along a path.
        
        Args:
            path: List of (x, y, t) tuples representing a robot's path
        """
        for x, y, t in path:
            self.add_reservation((x, y), t)
    
    def is_collision(self, position: Tuple[int, int], time: int) -> bool:
        """Check if position at time is occupied."""
        return (position, time) in self.reservations
    
    def heuristic(self, pos_a: Tuple[int, int], pos_b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1])
    
    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighboring nodes in the graph."""
        return list(self.graph.neighbors(node_id))
    
    def node_id_to_pos(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to position tuple."""
        return self.graph.nodes[node_id].get('pos', (0, 0))
    
    def pos_to_node_id(self, pos: Tuple[int, int]) -> Optional[int]:
        """Convert position tuple to node ID."""
        for node_id, data in self.graph.nodes(data=True):
            if data.get('pos') == pos:
                return node_id
        return None
    
    def find_path(
        self,
        start: Union[int, Tuple[int, int]],
        goal: Union[int, Tuple[int, int]],
        start_time: int = 0
    ) -> List[Tuple[int, int, int]]:
        """
        Find collision-free path from start to goal.
        
        Args:
            start: Start position (node ID or (x, y) tuple)
            goal: Goal position (node ID or (x, y) tuple)
            start_time: Time when robot starts moving
            
        Returns:
            List of (x, y, t) tuples representing the path, or empty list if no path found
        """
        # Convert to positions if needed
        if isinstance(start, int):
            start_pos = self.node_id_to_pos(start)
            start_node = start
        else:
            start_pos = start
            start_node = self.pos_to_node_id(start)
        
        if isinstance(goal, int):
            goal_pos = self.node_id_to_pos(goal)
            goal_node = goal
        else:
            goal_pos = goal
            goal_node = self.pos_to_node_id(goal)
        
        if start_node is None or goal_node is None:
            return []
        
        # Initialize search
        start_h = self.heuristic(start_pos, goal_pos)
        start_search_node = Node(
            f_score=start_h,
            g_score=0,
            position=start_pos,
            time=start_time,
            parent=None
        )
        
        open_set = [start_search_node]
        closed_set: Set[Tuple[Tuple[int, int], int]] = set()
        g_scores: Dict[Tuple[Tuple[int, int], int], float] = {
            (start_pos, start_time): 0
        }
        
        while open_set:
            current = heapq.heappop(open_set)
            
            # Check if reached goal
            if current.position == goal_pos:
                return self._reconstruct_path(current)
            
            # Skip if already processed
            state = (current.position, current.time)
            if state in closed_set:
                continue
            closed_set.add(state)
            
            # Check time limit
            if current.time >= self.max_time:
                continue
            
            # Get current node ID
            current_node_id = self.pos_to_node_id(current.position)
            if current_node_id is None:
                continue
            
            # Explore neighbors
            next_time = current.time + 1
            
            # Option 1: Move to neighbor
            for neighbor_id in self.get_neighbors(current_node_id):
                neighbor_pos = self.node_id_to_pos(neighbor_id)
                
                # Check collision
                if self.is_collision(neighbor_pos, next_time):
                    continue
                
                tentative_g = current.g_score + 1
                state = (neighbor_pos, next_time)
                
                if state not in g_scores or tentative_g < g_scores[state]:
                    g_scores[state] = tentative_g
                    h = self.heuristic(neighbor_pos, goal_pos)
                    f = tentative_g + h
                    
                    neighbor_node = Node(
                        f_score=f,
                        g_score=tentative_g,
                        position=neighbor_pos,
                        time=next_time,
                        parent=current
                    )
                    heapq.heappush(open_set, neighbor_node)
            
            # Option 2: Wait at current position
            if not self.is_collision(current.position, next_time):
                tentative_g = current.g_score + 1
                state = (current.position, next_time)
                
                if state not in g_scores or tentative_g < g_scores[state]:
                    g_scores[state] = tentative_g
                    h = self.heuristic(current.position, goal_pos)
                    f = tentative_g + h
                    
                    wait_node = Node(
                        f_score=f,
                        g_score=tentative_g,
                        position=current.position,
                        time=next_time,
                        parent=current
                    )
                    heapq.heappush(open_set, wait_node)
        
        # No path found
        return []
    
    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int, int]]:
        """Reconstruct path from goal node to start."""
        path = []
        current = node
        
        while current is not None:
            path.append((*current.position, current.time))
            current = current.parent
        
        path.reverse()
        return path


class PrioritizedPlanner:
    """
    Multi-robot planner using prioritized planning.
    
    Plans robots sequentially in priority order, with higher-priority robots
    planned first. Each robot uses space-time A* to avoid collisions with
    previously planned robots.
    """
    
    def __init__(self, graph: nx.Graph, max_time: int = 100):
        """
        Initialize prioritized planner.
        
        Args:
            graph: NetworkX graph representing the environment
            max_time: Maximum time horizon for planning
        """
        self.graph = graph
        self.max_time = max_time
    
    def plan(
        self,
        robots: Dict[str, Dict],
        use_priorities: bool = True
    ) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Plan paths for all robots.
        
        Args:
            robots: Dictionary mapping robot IDs to robot data with keys:
                   - 'start': start position (node ID or tuple)
                   - 'goal': goal position (node ID or tuple)
                   - 'priority': robot priority (higher = planned first)
                   - 'start_time': time when robot starts (default 0)
            use_priorities: Whether to use robot priorities for ordering
            
        Returns:
            Dictionary mapping robot IDs to paths (list of (x, y, t) tuples)
        """
        # Sort robots by priority (higher priority first)
        if use_priorities:
            sorted_robots = sorted(
                robots.items(),
                key=lambda x: x[1].get('priority', 1.0),
                reverse=True
            )
        else:
            sorted_robots = list(robots.items())
        
        # Initialize space-time A* planner
        planner = SpaceTimeAStar(self.graph, max_time=self.max_time)
        
        # Plan each robot sequentially
        paths = {}
        
        for robot_id, robot_data in sorted_robots:
            start = robot_data['start']
            goal = robot_data['goal']
            start_time = robot_data.get('start_time', 0)
            
            # Find path avoiding previous robots
            path = planner.find_path(start, goal, start_time)
            
            if not path:
                print(f"Warning: No path found for robot {robot_id}")
                paths[robot_id] = []
            else:
                paths[robot_id] = path
                # Add this robot's path to reservations
                planner.add_path_reservations(path)
        
        return paths


def run_prioritized_planning(
    problem,
    algorithm: str = 'astar',
    max_time: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Run prioritized planning on a PathfindingProblem.
    
    Args:
        problem: PathfindingProblem instance
        algorithm: Algorithm name (currently only 'astar' supported)
        max_time: Maximum time horizon (defaults to problem.T * 2)
        verbose: Print detailed solving information
        
    Returns:
        Dictionary mapping robot IDs to paths
    """
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
    
    # Prepare robot data
    robots_data = {}
    for robot_id, robot in problem.robots.items():
        # Convert start/goal to appropriate format
        if hasattr(problem, 'graph') and problem.graph is not None:
            if isinstance(robot.start, tuple):
                start = robot.start
                goal = robot.goal
            else:
                start = robot.start
                goal = robot.goal
        else:
            # For grid problems, use tuples
            if isinstance(robot.start, tuple):
                start = robot.start
                goal = robot.goal
            else:
                start = robot.start
                goal = robot.goal
        
        robots_data[robot_id] = {
            'start': start,
            'goal': goal,
            'priority': getattr(robot, 'priority', 1.0),
            'start_time': getattr(robot, 'start_time', 0)
        }
    
    # Set max time
    if max_time is None:
        max_time = problem.T * 2 if hasattr(problem, 'T') else 100
    
    # Run planner
    planner = PrioritizedPlanner(G, max_time=max_time)
    
    if verbose:
        # Sort to show planning order
        sorted_robots = sorted(
            robots_data.items(),
            key=lambda x: x[1].get('priority', 1.0),
            reverse=True
        )
        
        print("Planning robots sequentially:")
        for i, (robot_id, data) in enumerate(sorted_robots, 1):
            print(f"  [{i}/{len(sorted_robots)}] {robot_id}: {data['start']} → {data['goal']} (priority={data['priority']:.1f})")
    
    paths = planner.plan(robots_data, use_priorities=True)
    
    if verbose:
        print("\nResults:")
        for robot_id in sorted_robots:
            robot_id = robot_id[0]  # Extract ID from tuple
            if robot_id in paths and paths[robot_id]:
                print(f"  ✓ {robot_id}: {len(paths[robot_id])} steps")
            else:
                print(f"  ✗ {robot_id}: NO PATH")
    
    return paths
