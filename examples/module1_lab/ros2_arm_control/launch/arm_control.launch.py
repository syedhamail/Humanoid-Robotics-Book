from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for arm control nodes."""

    return LaunchDescription([
        Node(
            package='ros2_arm_control',
            executable='arm_controller',
            name='arm_controller',
            output='screen',
            parameters=[
                {'use_sim_time': True}  # Use simulation time if available
            ]
        ),
        Node(
            package='ros2_arm_control',
            executable='arm_client',
            name='arm_client',
            output='screen',
            parameters=[
                {'use_sim_time': True}  # Use simulation time if available
            ]
        )
    ])