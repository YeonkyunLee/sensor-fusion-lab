"""Pure-Python, ROS-free synthetic sensor generator for the fusion pipeline.

This module stands in for a Gazebo robot: given a simple analytic ground-truth
trajectory, it emits time-stamped **position** fixes (GPS/odometry-like, noisy)
and **IMU accelerometer** samples that match the exact message CONTRACT the
existing ``kalman_fusion`` node consumes:

* position  -> ``geometry_msgs/PointStamped``  (fields: ``stamp``, ``x``, ``y``)
              units: metres, in the ``map`` frame.
* imu       -> ``sensor_msgs/Imu``             (fields: ``stamp``, ``ax``, ``ay``)
              units: linear acceleration in m/s^2.

There are **no** ``rclpy`` and **no** ``gz`` imports here — only the standard
library plus NumPy. That is what lets the whole sim -> bridge -> fusion contract
be exercised in CI without Gazebo or a ROS2 install. The ROS node
(``mock_driver_node``) and the Gazebo/ros_gz_bridge path merely wrap the samples
this module produces into real ROS messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np


# --- message-contract sample types -----------------------------------------
@dataclass(frozen=True)
class PositionSample:
    """Synthetic ``geometry_msgs/PointStamped`` payload (metres, ``map`` frame)."""

    stamp: float  # seconds since start
    x: float
    y: float

    def as_point(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class ImuSample:
    """Synthetic ``sensor_msgs/Imu`` payload (linear acceleration, m/s^2)."""

    stamp: float  # seconds since start
    ax: float
    ay: float

    def as_accel(self) -> Tuple[float, float]:
        return (self.ax, self.ay)


# --- ground-truth analytic path --------------------------------------------
def true_trajectory(
    n: int, dt: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A smooth analytic 2D path (figure-eight): times, position, velocity, accel.

    This is the same style of curved trajectory a small differential-drive robot
    would trace in the Gazebo world. Position/velocity/acceleration are mutually
    consistent (velocity and accel are the analytic derivatives), so the accel
    stream is a faithful "IMU truth" for the position stream.
    """
    t = np.arange(n) * dt
    wx, wy = 0.3, 0.6
    Ax, Ay = 20.0, 12.0
    px = Ax * np.sin(wx * t)
    py = Ay * np.sin(wy * t)
    vx = Ax * wx * np.cos(wx * t)
    vy = Ay * wy * np.cos(wy * t)
    ax = -Ax * wx * wx * np.sin(wx * t)
    ay = -Ay * wy * wy * np.sin(wy * t)
    pos = np.stack([px, py], axis=1)
    vel = np.stack([vx, vy], axis=1)
    acc = np.stack([ax, ay], axis=1)
    return t, pos, vel, acc


class MockSensorSource:
    """Synthetic sensor source that mimics the Gazebo robot's two sensors.

    Produces a paired stream of noisy position fixes and noisy IMU accel
    samples along an analytic ground-truth path. Deterministic given ``seed``.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    dt : float
        Timestep between samples (s). Also the nominal sensor period.
    pos_sigma : float
        Std-dev of position measurement noise (m).
    imu_sigma : float
        Std-dev of IMU accel measurement noise (m/s^2).
    imu_bias : float
        Constant accelerometer bias added to every axis (m/s^2) — the source of
        integration drift that fusion with the position sensor corrects.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n: int = 300,
        dt: float = 0.1,
        pos_sigma: float = 2.0,
        imu_sigma: float = 0.5,
        imu_bias: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.n = int(n)
        self.dt = float(dt)
        self.pos_sigma = float(pos_sigma)
        self.imu_sigma = float(imu_sigma)
        self.imu_bias = float(imu_bias)
        self._rng = np.random.default_rng(int(seed))

        self.t, self.pos, self.vel, self.acc = true_trajectory(self.n, self.dt)
        self._z_pos = self.pos + self._rng.normal(0.0, self.pos_sigma, self.pos.shape)
        self._z_acc = (
            self.acc
            + self._rng.normal(0.0, self.imu_sigma, self.acc.shape)
            + self.imu_bias
        )

    # -- ground truth (for tests / evaluation) ------------------------------
    @property
    def ground_truth(self) -> np.ndarray:
        """The clean analytic position path ``[n, 2]`` (metres)."""
        return self.pos.copy()

    @property
    def noisy_positions(self) -> np.ndarray:
        """The raw noisy position measurements ``[n, 2]`` (metres)."""
        return self._z_pos.copy()

    @property
    def noisy_accels(self) -> np.ndarray:
        """The raw noisy accel measurements ``[n, 2]`` (m/s^2)."""
        return self._z_acc.copy()

    # -- sample accessors ---------------------------------------------------
    def position_sample(self, k: int) -> PositionSample:
        return PositionSample(
            stamp=float(self.t[k]), x=float(self._z_pos[k, 0]), y=float(self._z_pos[k, 1])
        )

    def imu_sample(self, k: int) -> ImuSample:
        return ImuSample(
            stamp=float(self.t[k]), ax=float(self._z_acc[k, 0]), ay=float(self._z_acc[k, 1])
        )

    def position_samples(self) -> List[PositionSample]:
        return [self.position_sample(k) for k in range(self.n)]

    def imu_samples(self) -> List[ImuSample]:
        return [self.imu_sample(k) for k in range(self.n)]

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[Tuple[ImuSample, PositionSample]]:
        """Iterate paired ``(imu_sample, position_sample)`` per timestep."""
        for k in range(self.n):
            yield self.imu_sample(k), self.position_sample(k)


def generate_stream(
    n: int = 300, dt: float = 0.1, **kwargs
) -> Tuple[List[ImuSample], List[PositionSample], np.ndarray]:
    """Convenience: return ``(imu_samples, position_samples, ground_truth)``."""
    src = MockSensorSource(n=n, dt=dt, **kwargs)
    return src.imu_samples(), src.position_samples(), src.ground_truth
