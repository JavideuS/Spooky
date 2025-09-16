import map


class PathfindingProblem:
    def __init__(self, start, end, grid=None, graph=None, T=None, name="unnamed"):
        # Support both grid and graph formats
        self.grid = grid
        self.graph = graph
        
        # Validate that at least one format is provided
        if self.grid is None and self.graph is None:
            raise ValueError("Either grid or graph must be provided")
        
        # Start and end can be coordinates (for grid) or node indices (for graph)
        self.start = start
        self.end = end
        
        if T is not None:
            self.T = T
        else:
            # Calculate T based on available format
            if self.grid is not None:
                self.T = int(self.manhattan_distance(start, end) * 1.5)
            else:  # graph format
                # For graphs, we can estimate based on shortest path or use a default
                self.T = self._estimate_graph_time_horizon()
        
        self.name = name  # Name for problem, maybe corridor, hall...

    @classmethod
    def from_grid_dict(cls, grid, problem_dict):
        """
        Create a PathfindingProblem instance from a grid and dictionary.
        It receives problem section from config file
        and extracts problem parameters.
        The grid is expected since you probably will also be extracting it from config previously.
        """
        start = tuple(problem_dict["start"])
        end = tuple(problem_dict["goal"])
        T = problem_dict.get("T", None)
        return cls(start, end, grid=grid, T=T)
    
    @classmethod
    def from_dict(cls, problem_dict):
        """
        Create a PathfindingProblem instance from a configuration dictionary.
        It receives problem section from config file
        and extracts grid parameters.
        """
        grid = map.Grid.from_dict(problem_dict)
        return cls.from_grid_dict(grid, problem_dict)
    
    @classmethod
    def from_graph_data(cls, graph_data, start_node, end_node, T=None, name="graph_problem"):
        """
        Create a PathfindingProblem instance from graph data.
        
        Args:
            graph_data: Dictionary with 'nodes' and 'edges' keys or Graph instance
            start_node: Starting node index
            end_node: Goal node index
            T: Time horizon (optional)
            name: Problem name
        """
        # Convert dict to Graph instance if needed
        if isinstance(graph_data, dict):
            graph = map.Graph.from_hdf5_data(graph_data, name)
        else:
            graph = graph_data
            
        return cls(start_node, end_node, graph=graph, T=T, name=name)

    def _estimate_graph_time_horizon(self):
        """Estimate time horizon for graph-based problems."""
        if self.graph is None:
            return 10  # Default fallback
        
        # Simple estimation: number of nodes or a reasonable default
        num_nodes = len(self.graph.get('nodes', []))
        if num_nodes > 0:
            return min(max(num_nodes // 2, 5), 20)  # Between 5 and 20
        return 10
    
    def manhattan_distance(self, start, end):
        """Calculate Manhattan distance for grid coordinates."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def get_format_type(self):
        """Return the format type: 'grid', 'graph', or 'both'."""
        if self.grid is not None and self.graph is not None:
            return 'both'
        elif self.grid is not None:
            return 'grid'
        else:
            return 'graph'

    def to_dict(self):
        """
        Convert the problem instance to a dictionary representation.
        """
        result = {
            "start": self.start,
            "goal": self.end,
            "T": self.T
        }
        
        if self.grid is not None:
            result["grid"] = self.grid.to_dict()

        if self.graph is not None:
            result["graph"] = self.graph
  
        return result
