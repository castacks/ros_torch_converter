from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package='ros_torch_converter',
                namespace='viz_debug',
                executable='nav_plotter',
                name='debug_plt_viz_node'
            )
        ]
    )