from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='enhanced_map',
            executable='map_publisher',
            name='map_publisher',
            parameters=[
                {'map_file': 'maps/synthetic/3x3/no_obs3x3_mix.h5'},
                {'frame_id': 'map'}
            ],
            output='screen'
        )
    ])
