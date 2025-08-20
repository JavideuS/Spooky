# map_to_grid_map.py
import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

def add_layer_to_grid_map(grid_map_msg, layer_name, data):
    if data.ndim != 2:
        raise ValueError("Data must be 2D")

    data = np.array(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    flat_data = data.flatten(order='C').tolist()

    array_msg = Float32MultiArray()

    # ✅ dim[0] = y (rows), dim[1] = x (cols)
    dim_y = MultiArrayDimension()
    dim_y.label = "column_index"
    dim_y.size = data.shape[0]  # M rows
    dim_y.stride = data.shape[0]

    dim_x = MultiArrayDimension()
    dim_x.label = "row_index"
    dim_x.size = data.shape[1]  # N cols
    dim_x.stride = 1

    array_msg.layout.dim = [dim_y, dim_x]
    array_msg.layout.data_offset = 0
    array_msg.data = flat_data

    grid_map_msg.layers.append(layer_name)
    grid_map_msg.data.append(array_msg)



def map_to_grid_map(your_map, frame_id="map"):
    """
    Convert your custom map to a grid_map_msgs/GridMap.
    Uses your helper methods: get_color(), get_material_name(), etc.
    """
    msg = GridMap()
    msg.header.frame_id = frame_id
    msg.header.stamp = your_map.get_clock().now().to_msg() if hasattr(your_map, 'get_clock') else your_map.node.get_clock().now().to_msg()

    resolution = 2.0  # getattr(your_map, 'resolution', 1.0)
    msg.info.length_x = your_map.N * resolution
    msg.info.length_y = your_map.M * resolution
    msg.info.resolution = resolution

    msg.info.pose = Pose()
    msg.info.pose.position.x = msg.info.length_x / 2.0 - resolution / 2.0
    msg.info.pose.position.y = msg.info.length_y / 2.0 - resolution / 2.0

    msg.info.pose.position.z = 0.0

    # For 2D maps, orientation is identity (quaternion: w=1)
    msg.info.pose.orientation.w = 1.0
    msg.info.pose.orientation.x = 0.0
    msg.info.pose.orientation.y = 0.0
    msg.info.pose.orientation.z = 0.0

    # === occupancy layer ===
    occupancy = np.zeros((your_map.M, your_map.N), dtype=np.float32)
    for i, j in your_map.obstacles:
        if 0 <= i < your_map.M and 0 <= j < your_map.N:
            occupancy[i, j] = 100.0  # occupied
    add_layer_to_grid_map(msg, "occupancy", occupancy)

    # === elevation layer ===
    if your_map.elevation is not None:
        elevation = np.array(your_map.elevation, dtype=np.float32)
        add_layer_to_grid_map(msg, "elevation", elevation)

    # === terrain layer (material indices) ===
    if your_map.terrain is not None:
        terrain = np.array(your_map.terrain, dtype=np.float32)
        add_layer_to_grid_map(msg, "terrain", terrain)

    # === color layer (ARGB32 packed float32) ===
    if your_map.terrain is not None and hasattr(your_map, 'materials_data'):
        try:
            color_img = np.zeros((your_map.M, your_map.N, 3), dtype=np.float32)
            for i in range(your_map.M):
                for j in range(your_map.N):
                    idx = int(your_map.terrain[i, j])
                    color_name = your_map.get_color(idx)
                    # Convert string or RGB tuple to (r,g,b) in 0-1
                    rgb = parse_color(color_name)
                    color_img[i, j] = rgb

            # Pack into ARGB32 as uint32 → cast to float32 (GridMap expects float32)
            r = (color_img[:, :, 0] * 255).astype(np.uint32)
            g = (color_img[:, :, 1] * 255).astype(np.uint32)
            b = (color_img[:, :, 2] * 255).astype(np.uint32)
            a = np.full_like(r, 255)
            argb32 = (a << 24) | (r << 16) | (g << 8) | b
            # color_layer = argb32.astype(np.float32)  # GridMap uses float32 for colors
            color_layer = argb32.view(np.float32)

            add_layer_to_grid_map(msg, "color", color_layer)
        except Exception as e:
            print(f"Warning: failed to generate color layer: {e}")

    return msg


def parse_color(color_input):
    """
    Convert color from string or tuple to (r, g, b) in [0,1]
    Supports: "red", (0.5, 0.5, 0.5), "#FF0000"
    """
    import matplotlib.colors as mcolors

    try:
        if isinstance(color_input, (list, tuple)) and len(color_input) == 3:
            return np.clip(np.array(color_input), 0, 1)
        elif isinstance(color_input, str):
            rgb = mcolors.to_rgb(color_input)
            return np.array(rgb)
        else:
            return np.array([1.0, 1.0, 1.0])  # white fallback
    except:
        return np.array([1.0, 1.0, 1.0])
