import h5py

def generate_map(
    base_elevation,
    obstacle_mask,
    traffic_flow,
    terrain_map,
    output_path,
    metadata={}
):
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('elevation', data=base_elevation)
        f.create_dataset('occupancy', data=obstacle_mask)
        f.create_dataset('traffic_flow', data=traffic_flow)
        f.create_dataset('terrain_type', data=terrain_map)
        for k, v in metadata.items():
            f.attrs[k] = v