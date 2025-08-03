import h5py
import numpy as np
from pathlib import Path

def load_map_from_hdf5(h5_path):
    """
    Load map data from HDF5 file and return structured dict.
    
    Returns:
        dict with keys:
            - 'name': str
            - 'grid': {'M': int, 'N': int, 'obstacles': list of tuples}
            - 'terrain': {'base_material': str, 'modifications': list} [optional]
            - 'elevation': {'base_height': float, 'modifications': list} [optional]
            - 'resolution': float
    """
    with h5py.File(h5_path, 'r') as f:
        # Load required data
        occupancy = f['map_structure'][:]
        M, N = occupancy.shape
        
        # Extract obstacles (where occupancy == 1)
        obstacles = []
        obs_positions = np.where(occupancy == 1)
        for i, j in zip(obs_positions[0], obs_positions[1]):
            obstacles.append((int(i), int(j)))

        # Load metadata
        map_name = f.attrs.get('map_name', Path(h5_path).stem)
        resolution = f.attrs.get('resolution', 1.0)

        # Load materials list (as dataset, not attribute)
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

        # Optional: Load terrain and elevation grids
        if 'terrain' in f:
            result['terrain_grid'] = f['terrain'][:]
        if 'elevation' in f:
            result['elevation_grid'] = f['elevation'][:]

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