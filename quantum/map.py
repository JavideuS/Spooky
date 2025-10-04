import numpy as np


class Grid:
    def __init__(self, M, N, obstacles=None, terrain=None, elevation=None,
                 materials=None, materials_data=None, resolution=1.0, name="unnamed"):
        self.M = M
        self.N = N
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity
        self.obstacles = obstacles or []
        self.terrain = terrain  # 2D numpy array of material indices (optional)
        self.elevation = elevation  # 2D numpy array of heights (optional)
        # List: index → material_name
        self.materials = materials if materials is not None else []
        self.materials_data = materials_data if materials_data is not None else {}
        self.adjacency = self.build_adjacency()
        self.name = name
        self.resolution = resolution  # meters per grid cell

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
        materials_data = grid_dict.get("materials_data", {})
        return cls(M, N, obstacles=obstacles, materials=materials,
                   materials_data=materials_data,
                   name=grid_dict.get("name", "unnamed"))
    
    @classmethod
    def from_hdf5_data(cls, map_data, materials_data=None, name=None):
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
        resolution = map_data.get('resolution', 1.0)
        
        return cls(
            M=M,
            N=N,
            obstacles=obstacles,
            terrain=terrain_grid,
            elevation=elevation_grid,
            materials=materials,
            materials_data=materials_data,
            resolution=resolution,
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

    def get_material_cost(self, index):
        """Get cost of material by index"""
        name = self.get_material_name(index)
        return self.materials_data[name].get("cost", 6.0)
    
    def get_color(self, index):
        """Get color of material by name"""
        name = self.get_material_name(index)
        return self.materials_data[name].get("color", "white")


class Graph:
    """Graph representation for pathfinding problems."""
    
    def __init__(self, nodes, edges, weights=None, name="unnamed"):
        """
        Initialize a graph with nodes and edges.
        
        Args:
            nodes: List of (x, y) coordinates or node data
            edges: List of (i, j) or (i, j, weight) tuples
            weights: Optional edge weights (if not provided in edges)
            name: Graph name
        """
        self.nodes = nodes
        self.edges = edges
        self.weights = weights or {}
        self.name = name
        self.adjacency = self._build_adjacency()
    
    def _build_adjacency(self):
        """Build adjacency list from edges."""
        adjacency = {}
        for i in range(len(self.nodes)):
            adjacency[i] = set()
        
        for edge in self.edges:
            if len(edge) == 2:
                i, j = edge
                i = int(i)
                j = int(j)
                weight = 1.0
            else:
                i, j, weight = edge
                i = int(i)
                j = int(j)
                weight = float(weight)
            
            if i < len(self.nodes) and j < len(self.nodes):
                adjacency[i].add((j, weight))
                # For undirected graphs, add both directions
                adjacency[j].add((i, weight))
        
        return adjacency
    
    @classmethod
    def from_hdf5_data(cls, graph_data, name=None):
        """Create Graph from HDF5-loaded data dict."""
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        resolution = graph_data.get('resolution', 1.0)
        
        return cls(
            nodes=nodes,
            edges=edges,
            name=name or graph_data.get('name', 'unnamed')
        )
    
    def get_node_position(self, node_id):
        """Get (x, y) position of a node."""
        if 0 <= node_id < len(self.nodes):
            return tuple(self.nodes[node_id])
        return None
    
    def get_node_from_position(self, position):
        """Get node index from (x, y) position."""
        for idx, pos in enumerate(self.nodes):
            if tuple(pos) == tuple(position):
                return idx
        return None
    
    def get_edge_weight(self, i, j):
        """Get weight of edge between nodes i and j."""
        for edge in self.edges:
            if len(edge) == 2:
                if (edge[0] == i and edge[1] == j) or (edge[0] == j and edge[1] == i):
                    return 1.0
            else:
                if (edge[0] == i and edge[1] == j) or (edge[0] == j and edge[1] == i):
                    return edge[2]
        return float('inf')  # No edge exists
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "name": self.name
        }
