"""Camera-only track perception test. Publishes no motion requests."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    config = str(Path(get_package_share_directory('team_navigation')) /
                 'config' / 'perception.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('namespace', description='Assigned robot namespace'),
        DeclareLaunchArgument('web_port', default_value='8081',
                              description='Browser debug view port; 0 disables it'),
        Node(package='team_navigation', executable='track_test',
             namespace=LaunchConfiguration('namespace'), output='screen',
             parameters=[config, {'web_port': ParameterValue(
                 LaunchConfiguration('web_port'), value_type=int)}]),
    ])
