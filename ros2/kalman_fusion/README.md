# kalman_fusion (ROS2 ament_python package)

A ROS2 node that wraps the repo's linear Kalman filter to fuse a **position**
sensor (GPS-like, noisy, may drop out) with an **IMU** (accelerometer, fast but
drifts) into a single published state estimate. Same estimation idea as
[`scripts/02_imu_fusion.py`](../../scripts/02_imu_fusion.py), packaged as a
buildable ROS2 node.

## Design: ROS-free core

The estimation lives in [`kalman_fusion/fusion_core.py`](kalman_fusion/fusion_core.py)
(`FusionCore`) — pure Python + NumPy, **no `rclpy`**. The node
([`kalman_fusion/fusion_node.py`](kalman_fusion/fusion_node.py)) imports `rclpy`
and the message types behind a guard, so the module can be imported (and the
core unit-tested) on a machine **without** a ROS2 install. `FusionCore` inlines
a minimal copy of `src/sensor_fusion/kalman.py` (attributed in the source) so
the package is self-contained and `colcon`-buildable.

## Topics

| Direction | Topic          | Type                        | Meaning                          |
| --------- | -------------- | --------------------------- | -------------------------------- |
| sub       | `/position`    | `geometry_msgs/PointStamped`| noisy position fix (GPS-like)    |
| sub       | `/imu`         | `sensor_msgs/Imu`           | linear acceleration              |
| pub       | `/fused_odom`  | `nav_msgs/Odometry`         | fused pose (x,y) + twist (vx,vy) |

Parameters: `pos_sigma` (default 2.0), `imu_sigma` (0.5), and the three topic
names (`position_topic`, `imu_topic`, `output_topic`).

## Build & run (requires a ROS2 environment)

```bash
# in a ROS2 workspace, with this package under src/
source /opt/ros/humble/setup.bash      # or jazzy
cd ~/ros2_ws
colcon build --packages-select kalman_fusion
source install/setup.bash

# run the node directly ...
ros2 run kalman_fusion fusion_node

# ... or via the launch file
ros2 launch kalman_fusion fusion.launch.py
```

Feed it data on `/position` and `/imu` (e.g. `ros2 topic pub`, a bag, or a
simulator) and echo the result:

```bash
ros2 topic echo /fused_odom
```

## Honest testing note

The **estimation core is CI-tested headless** — see
[`tests/test_ros2_core.py`](../../tests/test_ros2_core.py), which imports
`FusionCore` with no ROS2 present, runs a simulated position+IMU sequence, and
asserts the fused estimate tracks ground truth better than the raw position
measurements. It also asserts `kalman_fusion.fusion_node` imports cleanly
without `rclpy` (the guard works).

A **full end-to-end ROS2 spin is NOT exercised in CI** — that requires an actual
ROS2 install (Humble/Jazzy) with `rclpy`, `geometry_msgs`, `sensor_msgs`, and
`nav_msgs`. The `colcon build` / `ros2 run` steps above are the real,
documented path on such a machine; they are not run by the repo's headless test
suite.
