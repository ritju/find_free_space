
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

def get_environment_value(var, env, default):
    try:
        if env in os.environ:
            var[0] = os.environ.get(env)
            print(f'get {env} value: {var[0]} from docker_compose.yaml file')
        else:
            var[0] = default
            print(f"Using default {env} value: {default}.")
    except Exception as e:
        print(f'exception: {str(e)}')
        print(f"Please input {env} in docker_compose.yaml")
        var[0] = default

def generate_launch_description():
        search_radius_min = [3.0]
        search_radius_max = [4.0]
        outside_min = [0.0]
        outside_max = [0.5]

        get_environment_value(search_radius_min, 'GARAGE_FIND_PARKING_POINT_SEARCH_RADIUS_MIN', 3.0)
        get_environment_value(search_radius_max, 'GARAGE_FIND_PARKING_POINT_SEARCH_RADIUS_MAX', 4.0)
        get_environment_value(outside_min, 'GARAGE_FIND_PARKING_POINT_OUTSIDE_MIN', 0.0)
        get_environment_value(outside_max, 'GARAGE_FIND_PARKING_POINT_OUTSIDE_MAX', 0.5)
        
        return LaunchDescription([
                Node(
                        package='find_free_space',
                        executable='find_parking_space',
                        name='find_free_space_action_server',
                        output='screen',
                        respawn_delay=2.0,
                        parameters=[params_file,
                                    {'search_radius_min': search_radius_min[0],
                                    'search_radius_max': search_radius_max[0],
                                    'outside_min': outside_min[0],
                                    'outside_max': outside_max[0]}],
                ),
        ])