class Grid:
    def __init__(self, M, N, obstacles=None):
        self.M = M
        self.N = N
        # Define valid movements: 4-connectivity (no diagonals)
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.obstacles = obstacles or []
        self.adjacency = self.build_adjacency()

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
