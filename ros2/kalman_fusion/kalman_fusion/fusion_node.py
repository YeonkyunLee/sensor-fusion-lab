"""ROS2 node wrapping :class:`~kalman_fusion.fusion_core.FusionCore`.

Subscribes to a position topic and an IMU topic, runs the ROS-free
:class:`FusionCore`, and publishes the fused state estimate as odometry.

rclpy and the ROS message types are imported lazily (inside a guard) so this
module can be imported on a machine WITHOUT a ROS2 install — the estimation
core stays testable headless. Running the actual node still needs ROS2.

Topics
------
Subscribes:
  * ``/position``  (geometry_msgs/PointStamped) — noisy position fix (GPS-like)
  * ``/imu``       (sensor_msgs/Imu)            — linear acceleration
Publishes:
  * ``/fused_odom`` (nav_msgs/Odometry)         — fused pose + twist estimate
"""

from __future__ import annotations

# fusion_core is ROS-free, so this import is always safe.
from kalman_fusion.fusion_core import FusionCore

# --- ROS2 guard -------------------------------------------------------------
# Import rclpy and message types defensively. On a machine without ROS2 these
# fail, but the module (and FusionCore) must still import for headless tests.
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Imu

    _ROS2_AVAILABLE = True
    _ROS2_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only where ROS2 is absent
    _ROS2_AVAILABLE = False
    _ROS2_IMPORT_ERROR = exc
    Node = object  # so the class definition below still parses without ROS2


class FusionNode(Node):  # type: ignore[misc]
    """ROS2 node that fuses ``/position`` + ``/imu`` into ``/fused_odom``."""

    def __init__(self):
        super().__init__("kalman_fusion_node")

        # Parameters (overridable via ROS params / launch).
        self.declare_parameter("pos_sigma", 2.0)
        self.declare_parameter("imu_sigma", 0.5)
        self.declare_parameter("position_topic", "/position")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("output_topic", "/fused_odom")

        pos_sigma = self.get_parameter("pos_sigma").value
        imu_sigma = self.get_parameter("imu_sigma").value
        position_topic = self.get_parameter("position_topic").value
        imu_topic = self.get_parameter("imu_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.core = FusionCore(pos_sigma=pos_sigma, imu_sigma=imu_sigma)
        self._seeded = False
        self._last_stamp = None  # ROS time of previous predict, for dt

        self.pub = self.create_publisher(Odometry, output_topic, 10)
        self.create_subscription(PointStamped, position_topic, self._on_position, 10)
        self.create_subscription(Imu, imu_topic, self._on_imu, 10)

        self.get_logger().info(
            f"kalman_fusion up: sub {position_topic} + {imu_topic} -> pub {output_topic}"
        )

    # -- helpers -------------------------------------------------------------
    def _stamp_to_sec(self, stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def _predict_to(self, stamp) -> None:
        """Advance the filter to ``stamp`` using elapsed wall time as dt."""
        now = self._stamp_to_sec(stamp)
        if self._last_stamp is None:
            self._last_stamp = now
            return
        dt = now - self._last_stamp
        self._last_stamp = now
        if dt > 0.0:
            self.core.predict(dt)

    def _publish(self, stamp) -> None:
        st = self.core.state
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(st[0])
        msg.pose.pose.position.y = float(st[1])
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = float(st[2])
        msg.twist.twist.linear.y = float(st[3])
        self.pub.publish(msg)

    # -- callbacks -----------------------------------------------------------
    def _on_position(self, msg) -> None:
        z = [msg.point.x, msg.point.y]
        if not self._seeded:
            self.core.seed_position(z)
            self._seeded = True
        self._predict_to(msg.header.stamp)
        self.core.update_position(z)
        self._publish(msg.header.stamp)

    def _on_imu(self, msg) -> None:
        self._predict_to(msg.header.stamp)
        self.core.update_imu([msg.linear_acceleration.x, msg.linear_acceleration.y])
        self._publish(msg.header.stamp)


def main(args=None) -> int:
    """Console entry point. Exits cleanly with a message if ROS2 is unavailable."""
    if not _ROS2_AVAILABLE:
        print(
            "kalman_fusion: rclpy / ROS2 is not available in this environment,\n"
            "so the fusion node cannot spin here.\n"
            f"  import error: {_ROS2_IMPORT_ERROR}\n"
            "The estimation core (kalman_fusion.fusion_core.FusionCore) is ROS-free\n"
            "and is unit-tested headless. To run the node, build inside a ROS2\n"
            "(Humble/Jazzy) environment: `colcon build` then\n"
            "`ros2 run kalman_fusion fusion_node`."
        )
        return 1

    rclpy.init(args=args)
    node = FusionNode()
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
