"""Gazebo-sim mock pipeline — headless tests. 실행: pytest -q

Exercises the sim -> fusion CONTRACT with NO Gazebo and NO rclpy:

  * imports MockSensorSource (the ROS-free synthetic sensor core) from
    ros2/kalman_fusion_sim,
  * generates a synthetic position + IMU-accel sequence,
  * feeds it into the existing FusionCore (from ros2/kalman_fusion) exactly the
    way the ROS nodes would,
  * asserts the fused estimate tracks the mock ground-truth trajectory better
    than the raw noisy position (i.e. the end-to-end sim->fusion contract works
    headless),
  * asserts the sim's ROS nodes import cleanly behind their rclpy guards.

Fast (<15s): a few hundred timesteps of a 6-state Kalman filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "ros2" / "kalman_fusion"))
sys.path.insert(0, str(_ROOT / "ros2" / "kalman_fusion_sim"))

from kalman_fusion.fusion_core import FusionCore  # noqa: E402
from kalman_fusion_sim.mock_driver_core import (  # noqa: E402
    ImuSample,
    MockSensorSource,
    PositionSample,
    generate_stream,
    true_trajectory,
)


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_mock_core_contract_fields_and_units():
    """The synthetic samples must match the message contract the node consumes."""
    src = MockSensorSource(n=50, dt=0.1, seed=3)
    assert len(src) == 50

    imu = src.imu_sample(0)
    pos = src.position_sample(0)
    assert isinstance(imu, ImuSample)
    assert isinstance(pos, PositionSample)

    # PointStamped-like: stamp + planar x/y (metres).
    assert hasattr(pos, "stamp") and hasattr(pos, "x") and hasattr(pos, "y")
    # Imu-like: stamp + planar linear acceleration (m/s^2).
    assert hasattr(imu, "stamp") and hasattr(imu, "ax") and hasattr(imu, "ay")

    # timestamps advance by dt
    assert abs(src.imu_sample(1).stamp - src.imu_sample(0).stamp - 0.1) < 1e-9

    # all finite
    for k in range(len(src)):
        s = src.imu_sample(k)
        p = src.position_sample(k)
        assert np.isfinite([s.ax, s.ay, p.x, p.y]).all()


def test_sim_to_fusion_beats_raw_position():
    """End-to-end: MockSensorSource -> FusionCore tracks truth better than raw pos."""
    dt = 0.1
    n = 400
    src = MockSensorSource(
        n=n, dt=dt, pos_sigma=2.0, imu_sigma=0.5, imu_bias=0.05, seed=0
    )
    imu_samples, pos_samples, truth = generate_stream(
        n=n, dt=dt, pos_sigma=2.0, imu_sigma=0.5, imu_bias=0.05, seed=0
    )

    core = FusionCore(pos_sigma=2.0, imu_sigma=0.5)
    core.seed_position(pos_samples[0].as_point())

    outage = set(range(160, 220))  # position sensor drops out; IMU carries it
    fused = np.zeros((n, 2))
    raw_pos = np.zeros((n, 2))
    for k in range(n):
        core.predict(dt)
        core.update_imu(imu_samples[k].as_accel())
        if k not in outage:
            core.update_position(pos_samples[k].as_point())
        fused[k] = core.position
        raw_pos[k] = pos_samples[k].as_point()

    # sim -> fusion contract: fused beats the raw noisy position sensor
    assert _rmse(fused, truth) < _rmse(raw_pos, truth)
    assert np.all(np.isfinite(core.state))

    # covariance stays symmetric / positive-definite
    P = core.covariance
    assert np.allclose(P, P.T, atol=1e-8)
    assert np.all(np.linalg.eigvals(P) > 0)


def test_true_trajectory_is_self_consistent():
    """Analytic accel is the second derivative of the analytic position path."""
    dt = 0.01
    t, pos, vel, acc = true_trajectory(500, dt=dt)
    # numeric second derivative of position ~= analytic accel (interior points)
    num_acc = np.gradient(np.gradient(pos, dt, axis=0), dt, axis=0)
    assert _rmse(num_acc[5:-5], acc[5:-5]) < 1e-2


def test_sim_nodes_import_without_rclpy():
    """Both sim ROS nodes must import even where rclpy is missing (guards work)."""
    import kalman_fusion_sim.mock_driver_node as mdn
    import kalman_fusion_sim.odom_to_position_node as o2p

    assert hasattr(mdn, "main")
    assert hasattr(o2p, "main")
    # main() should return nonzero (not raise) when ROS2 is absent.
    if not mdn._ROS2_AVAILABLE:
        assert mdn.main() == 1
    if not o2p._ROS2_AVAILABLE:
        assert o2p.main() == 1
