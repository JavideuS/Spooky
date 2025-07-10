import map

class PathfindingProblem:
    def __init__(self, grid, start, end, T=None):
        self.grid = grid
        self.start = start  # (i, j)
        self.end = end      # (i, j)
        self.T = int(self.manhattan_distance() * 1.5)

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
        return cls(grid, start, end, T)
    
    @classmethod
    def from_dict(cls, problem_dict):
        """
        Create a PathfindingProblem instance from a configuration dictionary.
        It receives problem section from config file
        and extracts grid parameters.
        """
        grid = map.Grid.from_dict(problem_dict)
        return cls.from_grid_dict(grid, problem_dict)

    def manhattan_distance(self):
        return abs(self.start[0] - self.end[0]) + abs(self.start[1] - self.end[1])