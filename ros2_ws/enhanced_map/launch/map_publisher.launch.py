from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    map_file_arg = DeclareLaunchArgument(
        'map_file',
        default_value='maps/synthetic/3x3/no_obs3x3_mix.h5',
        description='Path to the map file (h5)'
    )

    materials_file_arg = DeclareLaunchArgument(
        'materials_file',
        default_value=PathJoinSubstitution([FindPackageShare('enhanced_map'), 'config', 'materials.yaml']),
        description='Path to the materials configuration file'
    )

    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='map',
        description='Frame ID for the map'
    )

    # Get the values of the arguments
    map_file = LaunchConfiguration('map_file')
    materials_file = LaunchConfiguration('materials_file')
    frame_id = LaunchConfiguration('frame_id')

    return LaunchDescription([
        map_file_arg,
        materials_file_arg,
        frame_id_arg,

        Node(
            package='enhanced_map',
            executable='map_publisher',
            name='map_publisher',
            parameters=[
                {'map_file': map_file},
                {'materials_file': materials_file},
                {'frame_id': frame_id}
            ],
            output='screen'
        )
    ])