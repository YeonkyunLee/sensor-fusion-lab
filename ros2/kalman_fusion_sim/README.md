# kalman_fusion_sim (ROS2 ament_python package)

A simulation front-end for the [`kalman_fusion`](../kalman_fusion) node. It
drives the fusion pipeline two ways:

- **Run path A — full Gazebo:** a modern **gz sim** (Gazebo Harmonic / Ionic)
  world with a differential-drive robot carrying an **IMU** and an **odometry**
  publisher. `ros_gz_bridge` maps the gz sensor topics into the ROS2 topics the
  fusion node consumes.
- **Run path B — no-Gazebo mock:** a pure-Python synthetic sensor source
  (`MockSensorSource`) replayed by a thin ROS2 node onto the same topics, so the
  whole pipeline is demonstrable with **only** `rclpy` — no Gazebo, no bridge.

The estimation itself is unchanged: this package only produces `/imu` and
`/position`; `kalman_fusion` does the fusing and publishes `/fused_odom`.

## Design: ROS-free, Gazebo-free core

The synthetic sensor generator lives in
[`kalman_fusion_sim/mock_driver_core.py`](kalman_fusion_sim/mock_driver_core.py)
(`MockSensorSource`) — pure Python + NumPy, **no `rclpy`, no `gz`**. Given a
smooth analytic ground-truth path it emits time-stamped, noisy **position** and
**IMU accel** samples that match the message CONTRACT the fusion node reads
(fields + units). The ROS nodes
([`mock_driver_node.py`](kalman_fusion_sim/mock_driver_node.py),
[`odom_to_position_node.py`](kalman_fusion_sim/odom_to_position_node.py)) import
`rclpy` behind a guard, so the modules import — and the core is unit-tested — on
a machine with **neither** Gazebo nor ROS2.

## Topics

| Producer                         | gz topic                        | gz type            | ROS topic     | ROS type                       |
| -------------------------------- | ------------------------------- | ------------------ | ------------- | ------------------------------ |
| gz IMU sensor                    | `/imu`                          | `gz.msgs.IMU`      | `/imu`        | `sensor_msgs/msg/Imu`          |
| gz OdometryPublisher             | `/model/fusion_bot/odometry`    | `gz.msgs.Odometry` | `/odom`       | `nav_msgs/msg/Odometry`        |
| `odom_to_position` relay         | —                               | —                  | `/position`   | `geometry_msgs/msg/PointStamped` |
| ROS -> gz (drive the robot)      | `/model/fusion_bot/cmd_vel`     | `gz.msgs.Twist`    | `/cmd_vel`    | `geometry_msgs/msg/Twist`      |
| `kalman_fusion` fusion_node      | —                               | —                  | `/fused_odom` | `nav_msgs/msg/Odometry`        |

Why the relay? `ros_gz_bridge` maps gz Odometry only to `nav_msgs/Odometry`
(there is no gz -> `geometry_msgs/PointStamped` converter), so
[`odom_to_position_node.py`](kalman_fusion_sim/odom_to_position_node.py)
extracts `pose.position.{x,y}` into the `/position` PointStamped the fusion node
subscribes to. On run path B the mock driver publishes `/position` directly, so
no relay is needed.

## Run path A — full Gazebo (needs Gazebo + ROS2 installed)

Requires **Gazebo Harmonic/Ionic** and **ROS2 Jazzy/Humble** with `ros_gz_sim`
and `ros_gz_bridge`. Build the two packages in a colcon workspace:

```bash
source /opt/ros/jazzy/setup.bash          # or humble
cd ~/ros2_ws
colcon build --packages-select kalman_fusion kalman_fusion_sim
source install/setup.bash

# gz sim + bridge + relay + fusion node (add rviz:=true for RViz2)
ros2 launch kalman_fusion_sim sim_fusion.launch.py
```

Drive the robot so the sensors produce interesting motion:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.5}, angular: {z: 0.3}}' -r 10

ros2 topic echo /fused_odom
```

## Run path B — no-Gazebo mock demo (needs only ROS2)

No Gazebo required. The mock driver replays synthetic `/imu` + `/position`
straight into the fusion node:

```bash
source /opt/ros/jazzy/setup.bash
# terminal 1
ros2 run kalman_fusion_sim mock_driver
# terminal 2
ros2 run kalman_fusion fusion_node
# terminal 3
ros2 topic echo /fused_odom
```

## Run path C — fully headless (no Gazebo, no ROS2)

The sim -> fusion contract runs in plain Python. This is what CI exercises:

```bash
python -m pytest -q tests/test_gazebo_mock.py
```

`tests/test_gazebo_mock.py` imports `MockSensorSource`, feeds its output into
`FusionCore`, and asserts the fused estimate tracks the ground-truth trajectory
better than the raw noisy position — including a position-sensor outage the IMU
must carry.

## Honest testing note

The **mock sensor core and the sim -> fusion contract are CI-tested headless**
(`tests/test_gazebo_mock.py`), and both sim nodes are asserted to import without
`rclpy`.

The **full Gazebo path (run path A) is NOT exercised in CI.** It needs a real
Gazebo Harmonic/Ionic + ROS2 install with `ros_gz_sim` / `ros_gz_bridge`; the
`colcon build` / `ros2 launch` steps above are the real, documented path on such
a machine. The SDF world, bridge config, and launch file are validated only for
well-formedness (XML/YAML/py syntax) by the headless suite, not run.
