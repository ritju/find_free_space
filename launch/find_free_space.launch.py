
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import LoadComposableNodes
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessStart

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

find_free_space_pkg = get_package_share_directory("find_free_space")

params_file = os.path.join(find_free_space_pkg, "params", "config.yaml")

def generate_launch_description():
        return LaunchDescription([
                Node(
                        package='find_free_space',
                        executable='find_parking_space',
                        name='find_free_space_action_server',
                        output='screen',
                        respawn_delay=2.0,
                        parameters=[params_file]
                ),
        ])