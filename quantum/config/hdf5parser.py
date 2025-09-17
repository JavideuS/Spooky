import h5py
import numpy as np
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

def load_both_from_hdf5(h5_source):
    """
    Load both map (grid) and graph data from HDF5 file. 
    This is useful for synthetic maps that support both approaches.
    
    Args:
        h5_source: HDF5 file path or file-like object
        
    Returns:
        dict: Contains both 'map_data' and 'graph_data' keys, plus metadata
    """
    with h5py.File(h5_source, 'r') as f:
        # Load common metadata
        map_name = f.attrs.get('map_name', None)
        
        # Only use filename stem if map_name not set AND source is a path
        if map_name is None:
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
        
        # Check what data is available
        has_map = 'map_structure' in f
        has_graph = 'graph' in f
        
        result = {
            'name': map_name,
            'resolution': resolution,
            'has_map': has_map,
            'has_graph': has_graph,
            'map_data': None,
            'graph_data': None
        }
        
        # Load map data if available
        if has_map:
            occupancy = f['map_structure'][:]
            M, N = occupancy.shape
            
            # Extract obstacles
            obstacles = []
            obs_positions = np.where(occupancy == 1)
            for i, j in zip(obs_positions[0], obs_positions[1]):
                obstacles.append((int(i), int(j)))

            # Load materials
            materials = []
            if 'materials' in f:
                materials = [mat.decode('utf-8') for mat in f['materials'][:]]

            map_data = {
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
                map_data['terrain_grid'] = f['terrain'][:]
            if 'elevation' in f:
                map_data['elevation_grid'] = f['elevation'][:]
                
            result['map_data'] = map_data
        
        # Load graph data if available
        if has_graph:
            nodes = f["graph/nodes"][:]   # shape (N, 2)
            edges = f["graph/edges"][:]   # shape (E, 3)

            graph_data = {
                'name': map_name,
                'edges': edges,
                'nodes': nodes,
                'resolution': resolution,
            }
            
            result['graph_data'] = graph_data

        return result

    
# Example usage
if __name__ == "__main__":
    import sys
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "quantum/maps/synthetic/3x3/no_obs3x3.h5"
    
    # Example 1: Load both map and graph data
    print("=== Loading both map and graph data ===")
    both_data = load_both_from_hdf5(h5_path)
    print(f"Name: {both_data['name']}")
    print(f"Has map data: {both_data['has_map']}")
    print(f"Has graph data: {both_data['has_graph']}")
    print(f"Resolution: {both_data['resolution']}")
    
    if both_data['map_data']:
        print("\nMap data:")
        print(f"Grid: {both_data['map_data']['grid']['M']}x{both_data['map_data']['grid']['N']}")
        print(f"Obstacles: {both_data['map_data']['grid']['obstacles']}")
        if 'terrain_grid' in both_data['map_data']:
            print("Terrain grid loaded (shape):", both_data['map_data']['terrain_grid'].shape)
        if 'elevation_grid' in both_data['map_data']:
            print("Elevation grid loaded (shape):", both_data['map_data']['elevation_grid'].shape)
    
    if both_data['graph_data']:
        print("\nGraph data:")
        print(f"Nodes shape: {both_data['graph_data']['nodes'].shape}")
        print(f"Edges shape: {both_data['graph_data']['edges'].shape}")
    
    # Example 2: Load only map data (backward compatibility)
    print("\n=== Loading only map data (backward compatibility) ===")
    map_data = load_map_from_hdf5(h5_path)
    print(f"Name: {map_data['name']}")
    print(f"Grid: {map_data['grid']['M']}x{map_data['grid']['N']}")
    print(f"Obstacles: {map_data['grid']['obstacles']}")
    print(f"Resolution: {map_data['resolution']}")
    
    # Example 3: Load only graph data (backward compatibility)
    print("\n=== Loading only graph data (backward compatibility) ===")
    graph_data = load_graph_from_hdf5(h5_path)
    print(f"Name: {graph_data['name']}")
    print(f"Nodes shape: {graph_data['nodes'].shape}")
    print(f"Edges shape: {graph_data['edges'].shape}")
    print(f"Resolution: {graph_data['resolution']}")