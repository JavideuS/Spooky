class Grid:
    def __init__(self, M, N, obstacles=None, name="unnamed"):
        self.M = M
        self.N = N
        # Define valid movements: 4-connectivity (no diagonals)
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.obstacles = obstacles or []
        self.adjacency = self.build_adjacency()
        self.name = name

    @classmethod
    def from_dict(cls, grid_dict):
        """
        Create a Grid instance from a dictionary (for config file setup).
        It receives problem section from config file
        and extracts grid parameters.
        """
        M = grid_dict["grid"]["M"]
        N = grid_dict["grid"]["N"]
        obstacles = grid_dict["grid"]["obstacles"]
        return cls(M, N, obstacles=obstacles, name=grid_dict.get("name", "unnamed"))
    
    def to_dict(self):
        """
        Convert the grid instance to a dictionary representation.
        """
        return {
            "M": self.M,
            "N": self.N,
            "obstacles": self.obstacles,
            "adjacency": self.adjacency,
        }

    def build_adjacency(self):
        adjacency = {}
        for i in range(self.M):
            for j in range(self.N):
                # Skip if the position is in obstacles
                # if (i, j) in self.obstacles:
                #     continue
                neighbors = []
                for di, dj in self.moves:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.M and 0 <= nj < self.N:
                        # Skip if the position is in obstacles
                        if (ni, nj) in self.obstacles:
                            continue
                        neighbors.append((ni, nj))
                adjacency[(i, j)] = neighbors
        return adjacency
