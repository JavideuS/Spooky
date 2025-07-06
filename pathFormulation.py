class PathfindingProblem:
    def __init__(self, grid, start, end, T=None):
        self.grid = grid
        self.start = start  # (i, j)
        self.end = end      # (i, j)
        self.T = int(self.manhattan_distance() * 1.5)

    def manhattan_distance(self):
        return abs(self.start[0] - self.end[0]) + abs(self.start[1] - self.end[1])