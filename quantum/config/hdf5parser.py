import h5py
import numpy as np
from pathlib import Path

# config/hdf5parser.py
import h5py
from pathlib import Path

def load_map_from_hdf5(h5_source):
    """
    Load map data from HDF5 file. Accepts:
        - string path (e.g., "map.h5")
        - file-like object (e.g., SpooledTemporaryFile)
    
    Returns structured dict.
    """
    with h5py.File(h5_source, 'r') as f:
        # Load required data
        occupancy = f['map_structure'][:]
        M, N = occupancy.shape
        
        # Extract obstacles
        obstacles = []
        obs_positions = np.where(occupancy == 1)
        for i, j in zip(obs_positions[0], obs_positions[1]):
            obstacles.append((int(i), int(j)))

        # Load metadata
        map_name = f.attrs.get('map_name', None)
        
        # Only use filename stem if map_name not set AND source is a path
        # Note that path stem was giving problem with fastApi
        if map_name is None:
            # Try to get filename only if h5_source is a string or has name
            try:
                if isinstance(h5_source, (str, Path)):
                    map_name = Path(h5_source).stem
                elif hasattr(h5_source, 'name') and isinstance(h5_source.name, str):
                    map_name = Path(h5_source.name).stem
                else:
                    map_name = "unknown_map"
            except Exception:
                map_name = "unknown_map"

        resolution = f.attrs.get('resolution', 1.0)

        # Load materials
        materials = []
        if 'materials' in f:
            materials = [mat.decode('utf-8') for mat in f['materials'][:]]

        result = {
            'name': map_name,
            'grid': {
                'M': M,
                'N': N,
                'obstacles': obstacles
            },
            'resolution': resolution,
            'materials': materials
        }

        if 'terrain' in f:
            result['terrain_grid'] = f['terrain'][:]
        if 'elevation' in f:
            result['elevation_grid'] = f['elevation'][:]

        return result

def load_graph_from_hdf5(h5_source):
    with h5py.File(h5_source, "r") as f:
        nodes = f["graph/nodes"][:]   # shape (N, 2)
        edges = f["graph/edges"][:]   # shape (E, 3)

        # Load metadata
        map_name = f.attrs.get('map_name', None)
        
        # Only use filename stem if map_name not set AND source is a path
        # Note that path stem was giving problem with fastApi
        if map_name is None:
            # Try to get filename only if h5_source is a string or has name
            try:
                if isinstance(h5_source, (str, Path)):
                    map_name = Path(h5_source).stem
                elif hasattr(h5_source, 'name') and isinstance(h5_source.name, str):
                    map_name = Path(h5_source.name).stem
                else:
                    map_name = "unknown_map"
            except Exception:
                map_name = "unknown_map"

        resolution = f.attrs.get('resolution', 1.0)

        result = {
            'name': map_name,
            'edges': edges,
            'nodes': nodes,
            'resolution': resolution,
        }

        return result

    
# Example usage
if __name__ == "__main__":
    import sys
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "maps/test.h5"
    map_data = load_map_from_hdf5(h5_path)
    
    print("Loaded map from HDF5:")
    print(f"Name: {map_data['name']}")
    print(f"Grid: {map_data['grid']['M']}x{map_data['grid']['N']}")
    print(f"Obstacles: {map_data['grid']['obstacles']}")
    print(f"Resolution: {map_data['resolution']}")
    
    if 'terrain_grid' in map_data:
        print("Terrain grid loaded (shape):", map_data['terrain_grid'].shape)
        print("Terrain modifications (first 5 cells):", map_data['terrain_grid'][:5, :5])
        print("Materials:", map_data['materials'])
    if 'elevation_grid' in map_data:
        print("Elevation grid loaded (shape):", map_data['elevation_grid'].shape)