from grid_map_msgs.msg import GridMap
from rclpy.node import Node
import rclpy
import numpy as np

# Your modules
import quantum.config.hdf5parser as h5parser
import quantum.config.parser as config_parser
from quantum.map import Grid
from .helper_functions import map_to_grid_map


class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')

        # Declare parameter for map file
        self.declare_parameter('map_file', 'maps/synthetic/3x3/no_obs3x3_elev.h5')
        self.declare_parameter('materials_file', 'config/materials.yaml')
        self.declare_parameter('frame_id', 'map')


        map_file = self.get_parameter('map_file').get_parameter_value().string_value
        materials_file = self.get_parameter('materials_file').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        print("Map_file", map_file)
        # Load materials config
        materials_data = config_parser.load_config(materials_file)["materials"]

        # Load map from HDF5
        map_conf = h5parser.load_map_from_hdf5(map_file)
        self.my_map = Grid.from_hdf5_data(map_conf, materials_data)

        self.my_map.node = self  # for clock access in map_to_grid_map

        # Publisher
        self.pub = self.create_publisher(GridMap, '/map', 10)
        self.timer = self.create_timer(2.0, self.publish_map)

        self.get_logger().info(f"MapPublisher node started. Publishing map from {map_file}")

    def publish_map(self):
        try:
            msg = map_to_grid_map(self.my_map, frame_id=self.frame_id)
            self.pub.publish(msg)
            self.get_logger().info("Published enhanced grid_map with color!", once=True)
        except Exception as e:
            self.get_logger().error(f"Failed to publish grid map: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
