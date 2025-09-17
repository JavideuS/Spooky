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

        if isinstance(start_node, (list, tuple)):
            start_node = graph.get_node_from_position(start_node)
        if isinstance(end_node, (list, tuple)):
            end_node = graph.get_node_from_position(end_node)
        
        # print(start_node, end_node)
            
        return cls(start_node, end_node, graph=graph, T=T, name=name)

    @classmethod
    def from_unified_data(cls, h5_source, start, end, materials_data=None, T=None, name=None):
        """
        Create a unified PathfindingProblem instance with both grid and graph data.
        This is the main function for loading synthetic maps that support both approaches.
        
        Args:
            h5_source: HDF5 file path or file-like object
            start: Start position (i,j) for grid or node_id for graph
            end: End position (i,j) for grid or node_id for graph
            materials_data: Optional materials data for Grid object
            T: Time horizon (optional)
            name: Problem name (optional, will use map name if not provided)
            
        Returns:
            PathfindingProblem: Unified problem with both grid and graph representations
        """
        from config.hdf5parser import load_both_from_hdf5
        
        # Load both data types
        data = load_both_from_hdf5(h5_source)
        
        # Use provided name or map name
        problem_name = name or data['name']
        
        # Create grid if available
        grid = None
        if data['has_map'] and data['map_data']:
            grid = map.Grid.from_hdf5_data(
                data['map_data'],
                materials_data=materials_data,
                name=problem_name
            )
        
        # Create graph if available
        graph = None
        if data['has_graph'] and data['graph_data']:
            graph = map.Graph.from_hdf5_data(
                data['graph_data'],
                name=problem_name
            )
        
        # Convert coordinates to node indices for graph if needed
        graph_start = start
        graph_end = end
        if graph is not None and isinstance(start, (list, tuple)):
            graph_start = graph.get_node_from_position(start)
            graph_end = graph.get_node_from_position(end)
        
        # Create unified problem
        problem = cls(
            start=start,  # Keep original start for grid
            end=end,      # Keep original end for grid
            grid=grid,
            graph=graph,
            T=T,
            name=problem_name
        )
        
        # Store graph-specific start/end for convenience
        if graph is not None:
            problem.graph_start = graph_start
            problem.graph_end = graph_end
        
        return problem

    def _estimate_graph_time_horizon(self):
        """Estimate time horizon for graph-based problems."""
        if self.graph is None:
            return 10  # Default fallback
        
        # Simple estimation: number of nodes or a reasonable default
        num_nodes = len(self.graph.nodes)
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
    
    def get_graph_start_end(self):
        """Get graph-specific start and end node indices."""
        if hasattr(self, 'graph_start') and hasattr(self, 'graph_end'):
            return self.graph_start, self.graph_end
        elif self.graph is not None:
            # Convert coordinates to node indices if not already done
            start_node = self.graph.get_node_from_position(self.start) if isinstance(self.start, (list, tuple)) else self.start
            end_node = self.graph.get_node_from_position(self.end) if isinstance(self.end, (list, tuple)) else self.end
            return start_node, end_node
        else:
            return None, None
    
    def get_grid_start_end(self):
        """Get grid-specific start and end coordinates."""
        if self.grid is not None:
            return self.start, self.end
        else:
            return None, None
    
    def can_use_grid(self):
        """Check if grid representation is available."""
        return self.grid is not None
    
    def can_use_graph(self):
        """Check if graph representation is available."""
        return self.graph is not None

    def as_grid_only(self):
        """Return a new problem instance restricted to the grid representation."""
        if self.grid is None:
            raise ValueError("Grid representation not available in this problem")
        return PathfindingProblem(
            start=self.start,
            end=self.end,
            grid=self.grid,
            graph=None,
            T=self.T,
            name=self.name,
        )

    def as_graph_only(self):
        """Return a new problem instance restricted to the graph representation."""
        if self.graph is None:
            raise ValueError("Graph representation not available in this problem")
        g_start, g_end = self.get_graph_start_end()
        return PathfindingProblem(
            start=g_start,
            end=g_end,
            grid=None,
            graph=self.graph,
            T=self.T,
            name=self.name,
        )

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
