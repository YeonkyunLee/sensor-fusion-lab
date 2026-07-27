"""Full Gazebo (gz sim) + fusion launch — run path A.

Brings up:
  1. gz sim with worlds/fusion_world.sdf (differential-drive robot + IMU +
     odometry publisher)
  2. ros_gz_bridge (config/bridge.yaml): gz topics  <->  ROS2 topics
  3. odom_to_position relay: /odom (nav_msgs/Odometry) -> /position (PointStamped)
  4. the existing kalman_fusion fusion_node: /position + /imu -> /fused_odom
  5. (optional) RViz2 with `rviz:=true`

Requires a real install of Gazebo Harmonic/Ionic + ROS2 (Humble/Jazzy) with
ros_gz_sim / ros_gz_bridge. This launch is NOT exercised in CI (see README).

    ros2 launch kalman_fusion_sim sim_fusion.launch.py
    ros2 launch kalman_fusion_sim sim_fusion.launch.py rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_sim = get_package_share_directory("kalman_fusion_sim")
    world_path = os.path.join(pkg_sim, "worlds", "fusion_world.sdf")
    bridge_config = os.path.join(pkg_sim, "config", "bridge.yaml")

    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="false", description="Launch RViz2 if true."
    )

    # 1. gz sim — via ros_gz_sim's gz_sim.launch.py include.
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r -v 3 {world_path}"}.items(),
    )

    # 2. ros_gz_bridge — gz <-> ROS2 per config/bridge.yaml.
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{"config_file": bridge_config}],
    )

    # 3. relay: nav_msgs/Odometry /odom -> geometry_msgs/PointStamped /position.
    odom_to_position = Node(
        package="kalman_fusion_sim",
        executable="odom_to_position",
        name="odom_to_position",
        output="screen",
    )

    # 4. the existing fusion node.
    fusion = Node(
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

    # 5. optional RViz2.
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription([rviz_arg, gz_sim, bridge, odom_to_position, fusion, rviz])
