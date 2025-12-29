import map
import numpy as np
from robotConfiguration import RobotConfig


class PathfindingProblem:

    def __init__(self, robots, grid=None, graph=None, T=None, name="unnamed"):
        # Support both grid and graph formats
        self.grid = grid
        self.graph = graph
        
        # Validate that at least one format is provided
        if self.grid is None and self.graph is None:
            raise ValueError("Either grid or graph must be provided")
    
        self.robots = {}
        if isinstance(robots, dict):
            self.robots = robots
        elif isinstance(robots, RobotConfig):
            self.robots[robots.robot_id] = robots
        elif isinstance(robots, list):
            for robot in robots:
                self.robots[robot.robot_id] = robot
        self.num_robots = len(self.robots)

        if T is None:
            T = self.calculate_timeline()
        else:
            # Set individual robot times if not already set (and take the provided T as default)
            for robot in self.robots.values():
                if robot.T is None:
                    robot.T = T
        self.T = T
        self.name = name
        
    @classmethod
    def general_init(cls, start, end, grid=None, graph=None, T=None, name="unnamed"):
        # In this case we simply create a default robot configuration for single robot
        robot = RobotConfig("Angie", start, end)
        
        return cls(robot, grid, graph, T, name)

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
        return cls.general_init(start, end, grid=grid, T=T)
    
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
            
        return cls.general_init(start_node, end_node, graph=graph, T=T, name=name)

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
        
        # Create unified problem
        problem = cls.general_init(
            start=start,  # Keep original start for grid
            end=end,      # Keep original end for grid
            grid=grid,
            graph=graph,
            T=T,
            name=problem_name
        )
        
        return problem
    
    def add_robot(self, robot: RobotConfig, keep_time=False):
        """Add a robot to the problem."""
        self.robots[robot.robot_id] = robot
        self.num_robots += 1
        if not keep_time:
            self.T = self.calculate_timeline()
    
    def manhattan_distance(self, start, end):
        """Calculate Manhattan distance for grid coordinates."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def euclidean_distance(self, start, end):
        """Calculate Euclidean distance for graph coordinates."""
        return np.sqrt((start[0] - end[0]) * (start[0] - end[0]) + (start[1] - end[1]) * (start[1] - end[1]))

    # def is_valid_move(self, robot, from_pos, to_pos):
    #     """Check if a move is valid."""
    #     if self.grid is not None:
    #         return self.grid.is_valid_move(robot, from_pos, to_pos)
    #     else:
    #         return self.graph.is_valid_move(robot, from_pos, to_pos)
    
    def set_robot_time(self):
        """Set time horizon T for each robot if not already set."""
        for robot in self.robots.values():
            if robot.T is None:
                if self.grid is not None:
                    robot.T = int(self.manhattan_distance(robot.current_position, robot.goal) * 1.5)  # Simple heuristic
                else:   # graph format
                    # For graphs, I need to implement some heuristic like straight line from start to node
                    # And make a conversion from like meters to time steps and some extra margin
                    robot.T = 10

    def calculate_timeline(self):
        total_time = 0
        self.set_robot_time()
        for robot in self.robots.values():
            final_robot_time = robot.start_time + robot.T
            # print(robot.robot_id, final_robot_time)
            if final_robot_time > total_time:
                total_time = final_robot_time
        return total_time

    def get_robot_per_timestep(self):
        """
        Get a dictiorinary mapping each robot to that global timestep
        If the robot is inactive for that timestep, it will not appear in the list
        """
        robot_per_timestep = {}
        for t in range(self.T):
            robot_per_timestep[t] = []
            for robot in self.robots.values():
                if robot.start_time <= t < robot.start_time + robot.T:
                    robot_per_timestep[t].append(robot.robot_id)
        return robot_per_timestep

    def get_robot_nums(self):
        """
        Get the numberr associated to each robot id
        This works when retrieving variables from the QUBO
        """
        robot_num = {}
        for idx, robot_id in enumerate(self.robots.keys()):
            robot_num[robot_id] = idx
        return robot_num
    
    def get_format_type(self):
        """Return the format type: 'grid', 'graph', or 'both'."""
        if self.grid is not None and self.graph is not None:
            return 'both'
        elif self.grid is not None:
            return 'grid'
        else:
            return 'graph'
    
    def get_graph_robot_current_goal(self, robot_id):
        """Get graph-specific current_position and goal node indices from a robot."""
        if self.graph is not None:
            # Convert coordinates to node indices if not already done
            robot = self.robots[robot_id]
            start_node = (robot.current_position if isinstance(robot.current_position, int)
                          else self.graph.get_node_from_position(robot.current_position))
            
            end_node = (robot.goal if isinstance(robot.goal, int)
                        else self.graph.get_node_from_position(robot.goal))
            return start_node, end_node
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
            robots=self.robots,
            grid=self.grid,
            graph=None,
            T=self.T,
            name=self.name,
        )

    def as_graph_only(self):
        """Return a new problem instance restricted to the graph representation."""
        if self.graph is None:
            raise ValueError("Graph representation not available in this problem")
        return PathfindingProblem(
            robots=self.robots,
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
            "name": self.name,
            "T": self.T,
            "robots": {robot_id: robot.to_dict() for robot_id, robot in self.robots.items()},
        }

        if self.grid is not None:
            result["grid"] = self.grid.to_dict()

        if self.graph is not None:
            result["graph"] = self.graph.to_dict()
  
        return result
