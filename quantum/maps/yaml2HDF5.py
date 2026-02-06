import os
import yaml
import h5py
import numpy as np
from pathlib import Path

# Load materials mapping
def load_materials(config_path="config/materials.yaml"):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    material_to_id = {mat: info['index'] for mat, info in data['materials'].items()}
    return material_to_id

def apply_modifications(M, N, base_value, modifications, value_map=None):
    """
    Generic function to apply modifications to a grid.
    
    Args:
        M, N: grid dimensions
        base_value: default value (e.g., base material ID or base height)
        modifications: list of modification dicts
        value_map: optional dict to map string values to numeric (e.g., material names → IDs)
    """
    grid = np.full((M, N), base_value, dtype=np.float32)
    
    for mod in modifications:
        # Handle both 'height' (elevation) and 'type' (terrain)
        if 'height' in mod:
            value = mod['height']
        elif 'type' in mod:
            value = mod['type']
        else:
            value = base_value

        # Map string values (like material names) to numeric IDs
        if value_map and isinstance(value, str):
            value = value_map.get(value, value)

        # Apply to positions
        if 'positions' in mod:
            for x, y in mod['positions']:
                if 0 <= x < M and 0 <= y < N:
                    grid[x, y] = value

        # Apply to regions
        if 'region' in mod:
            start = mod['region']['start']
            end = mod['region']['end']
            for x in range(start[0], end[0] + 1):
                for y in range(start[1], end[1] + 1):
                    if 0 <= x < M and 0 <= y < N:
                        grid[x, y] = value

    return grid

def grid_to_graph_edges(occupancy, connectivity=4):
    """
    Convert an occupancy grid into a node graph (nodes + edges).
    
    Args:
        occupancy: 2D numpy array, 0 = free, 1 = obstacle
        connectivity: 4 or 8 (neighbors)
    
    Returns:
        nodes: np.ndarray of shape (N, 2), list of (x, y) positions
        edges: np.ndarray of shape (E, 3), list of (node_i, node_j, weight)
    """
    M, N = occupancy.shape
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 4-connectivity
    if connectivity == 8:
        moves += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Build node index map
    free_positions = [(x, y) for x in range(M) for y in range(N) if occupancy[x, y] == 0]
    node_to_index = {pos: i for i, pos in enumerate(free_positions)}
    nodes = np.array(free_positions, dtype=np.int32)

    # Build edges
    edges = []
    for (x, y) in free_positions:
        i = node_to_index[(x, y)]
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (nx, ny) in node_to_index:  # neighbor is free
                j = node_to_index[(nx, ny)]
                weight = 1.0 if abs(dx) + abs(dy) == 1 else np.sqrt(2)  # diagonals √2
                edges.append((i, j, weight))

    # Defining the structured dtype
    # (To avoid storing nodes as float too)
    dtype = [('i', np.int32), ('j', np.int32), ('weight', np.float32)]

    edges = np.array(edges, dtype=dtype)
    return nodes, edges

# Main function
def generate_map_from_yaml(yaml_path, output_dir="maps", materials_path="config/materials.yaml"):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    map_config = config['map']
    map_name = map_config['name']
    M = map_config['grid']['M']
    N = map_config['grid']['N']
    obstacles = map_config['grid'].get('obstacles', [])

    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, f"{map_name}.h5")

    # === Required: map_structure (occupancy) ===
    occupancy = np.zeros((M, N), dtype=np.uint8)
    for x, y in obstacles:
        if 0 <= x < M and 0 <= y < N:
            occupancy[x, y] = 1  # 1 = obstacle

    # === Optional layers (terrain, elevation) ===
    material_to_id = {}
    mat_config = {}
    if os.path.exists(materials_path):
        with open(materials_path, 'r') as f:
            mat_config = yaml.safe_load(f)
        material_to_id = {mat: info['index'] for mat, info in mat_config['materials'].items()}

    terrain_grid, local_materials_list = None, []
    if 'terrain' in map_config:
        terrain = map_config['terrain']
        base_mat = terrain['base_material']
        base_id = material_to_id.get(base_mat, 0)
        mods = terrain.get('modifications', [])

        terrain_grid_global = apply_modifications(M, N, base_id, mods, material_to_id).astype(np.uint8)
        unique_global_ids = np.unique(terrain_grid_global)
        global_to_local = {gid: lid for lid, gid in enumerate(sorted(unique_global_ids))}
        terrain_grid = np.vectorize(global_to_local.get)(terrain_grid_global)

        id_to_material = {info['index']: mat for mat, info in mat_config['materials'].items()}
        local_materials_list = [id_to_material[gid] for gid in sorted(unique_global_ids)]

    elevation_grid = None
    if 'elevation' in map_config:
        elev = map_config['elevation']
        base_height = elev.get('base_height', 0.0)
        mods = elev.get('modifications', [])
        elevation_grid = apply_modifications(M, N, base_height, mods)

    # === Build node graph from occupancy ===
    nodes, edges = grid_to_graph_edges(occupancy, connectivity=4)

    # === Save to HDF5 ===
    with h5py.File(h5_path, 'w') as f:
        # Grids
        f.create_dataset('map_structure', data=occupancy, compression='gzip')
        if terrain_grid is not None:
            f.create_dataset('terrain', data=terrain_grid, compression='gzip')
        if elevation_grid is not None:
            f.create_dataset('elevation', data=elevation_grid, compression='gzip')
        if local_materials_list:
            f.create_dataset('materials', data=[mat.encode('utf-8') for mat in local_materials_list])

        # Graph representation
        grp = f.create_group("graph")
        grp.create_dataset("nodes", data=nodes, compression='gzip')      # shape (N, 2)
        grp.create_dataset("edges", data=edges, compression='gzip')      # shape (E, 3)

        # Metadata
        f.attrs['map_name'] = map_name
        f.attrs['grid_size'] = f"{M}x{N}"
        f.attrs['resolution'] = map_config.get('resolution', 1.0)
        f.attrs['generated_from'] = os.path.relpath(yaml_path)
        f.attrs['generated_at'] = np.bytes_(str(np.datetime64('now')))

    print(f"✅ Generated: {h5_path}")


# Run on all YAML files
def generate_all_maps(scenarios_dir="scenarios", output_dir="maps"):
    for yaml_file in Path(scenarios_dir).glob("*.yaml"):
        generate_map_from_yaml(yaml_file, output_dir=output_dir)


if __name__ == "__main__":
    # generate_map_from_yaml(
    #     yaml_path="maps/synthetic/3x2/no_obs3x2.yaml",
    #     output_dir="maps/synthetic/3x2"
    # )
    generate_all_maps(scenarios_dir="maps/synthetic/3x3/", output_dir="maps/synthetic/3x3/")
