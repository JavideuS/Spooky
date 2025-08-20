from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=[FindPackageShare('quantum_planner_server'), '/config/quantum_goal.yaml']
        ),
        Node(
            package='quantum_planner_server',
            executable='quantum_client',
            name='ClientNode',
            parameters=[params_file]
        )
    ])