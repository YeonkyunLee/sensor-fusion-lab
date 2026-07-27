"""Relay node: nav_msgs/Odometry (/odom)  ->  geometry_msgs/PointStamped (/position).

The Gazebo OdometryPublisher (bridged by ros_gz_bridge to ``/odom`` as
``nav_msgs/msg/Odometry``) provides the robot's pose, but the ``kalman_fusion``
node subscribes to ``/position`` as ``geometry_msgs/msg/PointStamped``. There is
no gz -> PointStamped bridge converter, so this thin node performs the field
extraction (pose.position.x/y -> point.x/y), preserving the header/stamp.

``rclpy`` is imported behind a guard so this module imports on a machine WITHOUT
ROS2 (kept headless-importable). The conversion itself is trivial and stateless.
"""

from __future__ import annotations

# --- ROS2 guard -------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    from nav_msgs.msg import Odometry

    _ROS2_AVAILABLE = True
    _ROS2_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only where ROS2 is absent
    _ROS2_AVAILABLE = False
    _ROS2_IMPORT_ERROR = exc
    Node = object  # so the class definition below still parses without ROS2


class OdomToPositionNode(Node):  # type: ignore[misc]
    """Republish the translation of ``/odom`` as a ``/position`` PointStamped."""

    def __init__(self):
        super().__init__("odom_to_position")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("position_topic", "/position")

        self.pub = self.create_publisher(
            PointStamped, self.get_parameter("position_topic").value, 10
        )
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value, self._on_odom, 10
        )
        self.get_logger().info(
            f"odom_to_position up: {self.get_parameter('odom_topic').value} -> "
            f"{self.get_parameter('position_topic').value}"
        )

    def _on_odom(self, msg) -> None:
        out = PointStamped()
        out.header = msg.header
        out.point.x = msg.pose.pose.position.x
        out.point.y = msg.pose.pose.position.y
        out.point.z = msg.pose.pose.position.z
        self.pub.publish(out)


def main(args=None) -> int:
    """Console entry point. Exits cleanly with a message if ROS2 is unavailable."""
    if not _ROS2_AVAILABLE:
        print(
            "kalman_fusion_sim: rclpy / ROS2 is not available in this environment,\n"
            "so the odom_to_position relay cannot spin here.\n"
            f"  import error: {_ROS2_IMPORT_ERROR}\n"
            "This relay is only needed on the full Gazebo path (run path A)."
        )
        return 1

    rclpy.init(args=args)
    node = OdomToPositionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
