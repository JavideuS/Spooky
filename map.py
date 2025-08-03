import numpy as np


class Grid:
    def __init__(self, M, N, obstacles=None, terrain=None, elevation=None, materials=None, name="unnamed"):
        self.M = M
        self.N = N
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity
        self.obstacles = obstacles or []
        self.terrain = terrain  # 2D numpy array of material indices (optional)
        self.elevation = elevation  # 2D numpy array of heights (optional)
        # List: index → material_name
        self.materials = materials if materials is not None else []
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
        materials = grid_dict.get("materials", [])
        return cls(M, N, obstacles=obstacles, materials=materials,
                   name=grid_dict.get("name", "unnamed"))
    
    @classmethod
    def from_hdf5_data(cls, map_data, name=None):
        """
        Create Grid from HDF5-loaded data dict.
        
        Args:
            map_data: dict from load_map_from_hdf5()
        """
        M = map_data['grid']['M']
        N = map_data['grid']['N']
        obstacles = map_data['grid']['obstacles']
        
        # Optional layers
        terrain_grid = map_data.get('terrain_grid')
        elevation_grid = map_data.get('elevation_grid')
        materials = map_data.get('materials', [])  # if you pass material list
        
        return cls(
            M=M,
            N=N,
            obstacles=obstacles,
            terrain=terrain_grid,
            elevation=elevation_grid,
            materials=materials,
            name=name or map_data.get('name', 'unnamed')
        )
    
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

    def get_terrain_at(self, i, j):
        """Get material index at cell (i,j)"""
        if self.terrain is not None:
            return self.terrain[i, j]
        return None

    def get_elevation_at(self, i, j):
        """Get elevation at cell (i,j)"""
        if self.elevation is not None:
            return self.elevation[i, j]
        return None

    def get_material_name(self, index):
        """Convert material index to name"""
        if self.materials is None or len(self.materials) == 0:
            return "unknown"
        if 0 <= index < len(self.materials):
            return self.materials[index]
        return "unknown"

    def get_unique_materials_in_map(self):
        """Get list of material names actually present in this map"""
        if self.terrain is not None:
            unique_indices = np.unique(self.terrain)
            return [self.get_material_name(idx) for idx in unique_indices]
        return []

    def get_material_cost(self, index, material_costs):
        """Get cost of material by index"""
        name = self.get_material_name(index)
        return material_costs.get(name, 1.0)