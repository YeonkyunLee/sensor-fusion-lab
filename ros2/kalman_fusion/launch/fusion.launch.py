"""Launch the kalman_fusion node.

Run with:  ros2 launch kalman_fusion fusion.launch.py
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="kalman_fusion",
                executable="fusion_node",
                name="kalman_fusion_node",
                output="screen",
                parameters=[
                    {
                        "pos_sigma": 2.0,
                        "imu_sigma": 0.5,
                        "position_topic": "/position",
                        "imu_topic": "/imu",
                        "output_topic": "/fused_odom",
                    }
                ],
            )
        ]
    )
