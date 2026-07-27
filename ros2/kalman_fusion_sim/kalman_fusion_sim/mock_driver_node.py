"""ROS2 node that replays :class:`MockSensorSource` onto ``/imu`` + ``/position``.

This is the **no-Gazebo** demo driver: it publishes the synthetic sensor stream
from :mod:`kalman_fusion_sim.mock_driver_core` on exactly the topics the
existing ``kalman_fusion`` fusion node subscribes to, so the full
sim -> fusion pipeline can be driven with just ``rclpy`` — no Gazebo, no
ros_gz_bridge.

    ros2 run kalman_fusion_sim mock_driver

``rclpy`` and the ROS message types are imported behind a guard, so this module
imports cleanly on a machine WITHOUT a ROS2 install (the mock core stays
headless-testable). Running the node still needs ROS2.

Topics
------
Publishes:
  * ``/position``  (geometry_msgs/PointStamped) — noisy position fix
  * ``/imu``       (sensor_msgs/Imu)            — noisy linear acceleration
"""

from __future__ import annotations

# mock_driver_core is ROS-free, so this import is always safe.
from kalman_fusion_sim.mock_driver_core import MockSensorSource

# --- ROS2 guard -------------------------------------------------------------
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    from sensor_msgs.msg import Imu

    _ROS2_AVAILABLE = True
    _ROS2_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only where ROS2 is absent
    _ROS2_AVAILABLE = False
    _ROS2_IMPORT_ERROR = exc
    Node = object  # so the class definition below still parses without ROS2


class MockDriverNode(Node):  # type: ignore[misc]
    """Publishes synthetic ``/imu`` + ``/position`` from :class:`MockSensorSource`."""

    def __init__(self):
        super().__init__("kalman_fusion_mock_driver")

        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("n", 600)
        self.declare_parameter("pos_sigma", 2.0)
        self.declare_parameter("imu_sigma", 0.5)
        self.declare_parameter("imu_bias", 0.05)
        self.declare_parameter("seed", 0)
        self.declare_parameter("position_topic", "/position")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("frame_id", "map")

        rate_hz = float(self.get_parameter("rate_hz").value)
        n = int(self.get_parameter("n").value)
        self._frame_id = self.get_parameter("frame_id").value

        self.source = MockSensorSource(
            n=n,
            dt=1.0 / rate_hz,
            pos_sigma=float(self.get_parameter("pos_sigma").value),
            imu_sigma=float(self.get_parameter("imu_sigma").value),
            imu_bias=float(self.get_parameter("imu_bias").value),
            seed=int(self.get_parameter("seed").value),
        )

        self.pos_pub = self.create_publisher(
            PointStamped, self.get_parameter("position_topic").value, 10
        )
        self.imu_pub = self.create_publisher(
            Imu, self.get_parameter("imu_topic").value, 10
        )

        self._k = 0
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f"mock_driver up: publishing {n} samples at {rate_hz} Hz -> "
            f"{self.get_parameter('imu_topic').value} + "
            f"{self.get_parameter('position_topic').value}"
        )

    def _now_stamp(self):
        return self.get_clock().now().to_msg()

    def _tick(self) -> None:
        if self._k >= len(self.source):
            self.get_logger().info("mock_driver: stream complete")
            self.timer.cancel()
            return

        stamp = self._now_stamp()
        imu_s = self.source.imu_sample(self._k)
        pos_s = self.source.position_sample(self._k)

        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = self._frame_id
        imu_msg.linear_acceleration.x = imu_s.ax
        imu_msg.linear_acceleration.y = imu_s.ay
        imu_msg.linear_acceleration.z = 0.0
        imu_msg.orientation.w = 1.0
        self.imu_pub.publish(imu_msg)

        pos_msg = PointStamped()
        pos_msg.header.stamp = stamp
        pos_msg.header.frame_id = self._frame_id
        pos_msg.point.x = pos_s.x
        pos_msg.point.y = pos_s.y
        pos_msg.point.z = 0.0
        self.pos_pub.publish(pos_msg)

        self._k += 1


def main(args=None) -> int:
    """Console entry point. Exits cleanly with a message if ROS2 is unavailable."""
    if not _ROS2_AVAILABLE:
        print(
            "kalman_fusion_sim: rclpy / ROS2 is not available in this environment,\n"
            "so the mock driver node cannot spin here.\n"
            f"  import error: {_ROS2_IMPORT_ERROR}\n"
            "The synthetic sensor core (kalman_fusion_sim.mock_driver_core."
            "MockSensorSource)\nis ROS-free and is unit-tested headless. To run the "
            "driver, build inside a\nROS2 (Humble/Jazzy) environment: `colcon build` "
            "then\n`ros2 run kalman_fusion_sim mock_driver`."
        )
        return 1

    rclpy.init(args=args)
    node = MockDriverNode()
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
