"""Pure-Python, ROS-free estimation core for the fusion node.

This module holds a constant-acceleration Kalman filter that fuses a position
sensor (GPS-like, noisy, may drop out) with an IMU (accelerometer, fast but
drifts on its own). It is the exact same estimation idea as
``scripts/02_imu_fusion.py`` in this repo, packaged so it can be unit-tested
WITHOUT a ROS2 / rclpy install.

The ``KalmanFilter`` below is a minimal vendored copy of
``src/sensor_fusion/kalman.py`` (generic linear KF with multi-sensor update),
inlined so the ROS2 package is self-contained and ``colcon``-buildable without
reaching outside the package tree. No rclpy imports live here.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Discrete-time linear Kalman filter (vendored from src/sensor_fusion/kalman.py).

    state:  x_k = F x_{k-1} + w,   w ~ N(0, Q)
    meas:   z_k = H x_k + v,       v ~ N(0, R)
    """

    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.asarray(F, float)
        self.H = np.asarray(H, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.x = np.asarray(x0, float).reshape(-1)
        self.P = np.asarray(P0, float)
        self._I = np.eye(self.P.shape[0])

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, H=None, R=None) -> None:
        """Measurement update. Pass H/R for a specific sensor model (multi-sensor fusion)."""
        H = self.H if H is None else np.asarray(H, float)
        R = self.R if R is None else np.asarray(R, float)
        z = np.asarray(z, float).reshape(-1)
        y = z - H @ self.x  # innovation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        # Joseph form: keeps covariance symmetric / positive-definite numerically
        A = self._I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T


class FusionCore:
    """Position + IMU fusion using a constant-acceleration Kalman filter.

    State is ``[px, py, vx, vy, ax, ay]``. The transition matrix depends on the
    integration step ``dt``, so it is (re)built on every :meth:`predict`.

    Typical loop::

        core = FusionCore()
        core.predict(dt)            # time update
        core.update_imu([ax, ay])   # IMU is available every step
        core.update_position([x, y])  # position sensor when not in outage
        est = core.state            # fused [px, py, vx, vy, ax, ay]
    """

    def __init__(
        self,
        pos_sigma: float = 2.0,
        imu_sigma: float = 0.5,
        jerk_q: float = 0.5,
        x0=None,
        p0_scale: float = 10.0,
    ):
        self.pos_sigma = float(pos_sigma)
        self.imu_sigma = float(imu_sigma)
        self.jerk_q = float(jerk_q)

        # Measurement models: position observes [px,py]; IMU observes [ax,ay].
        self.H_pos = np.zeros((2, 6))
        self.H_pos[0, 0] = self.H_pos[1, 1] = 1.0
        self.H_imu = np.zeros((2, 6))
        self.H_imu[0, 4] = self.H_imu[1, 5] = 1.0
        self.R_pos = (self.pos_sigma**2) * np.eye(2)
        self.R_imu = (self.imu_sigma**2) * np.eye(2)

        x_init = np.zeros(6) if x0 is None else np.asarray(x0, float).reshape(-1)
        self._kf = KalmanFilter(
            F=np.eye(6),
            H=self.H_pos,
            Q=self._make_Q(0.1),
            R=self.R_pos,
            x0=x_init,
            P0=p0_scale * np.eye(6),
        )
        self._last_dt = 0.1

    def _make_F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[2, 4] = F[3, 5] = dt
        F[0, 4] = F[1, 5] = 0.5 * dt**2
        return F

    def _make_Q(self, dt: float) -> np.ndarray:
        Q = np.eye(6) * 1e-3
        # Model jerk (change in acceleration) as process noise on the accel states.
        Q[4, 4] = Q[5, 5] = self.jerk_q
        return Q

    def predict(self, dt: float) -> None:
        """Time update for step ``dt`` (rebuilds the constant-acceleration model)."""
        dt = float(dt)
        if dt <= 0.0:
            dt = self._last_dt
        self._last_dt = dt
        self._kf.F = self._make_F(dt)
        self._kf.Q = self._make_Q(dt)
        self._kf.predict()

    def update_position(self, z) -> None:
        """Fuse a 2D position measurement ``[x, y]`` (GPS-like sensor)."""
        self._kf.update(np.asarray(z, float).reshape(-1)[:2], H=self.H_pos, R=self.R_pos)

    def update_imu(self, accel) -> None:
        """Fuse a 2D acceleration measurement ``[ax, ay]`` (IMU)."""
        self._kf.update(np.asarray(accel, float).reshape(-1)[:2], H=self.H_imu, R=self.R_imu)

    def seed_position(self, z) -> None:
        """Initialise the position states directly (e.g. from the first fix)."""
        z = np.asarray(z, float).reshape(-1)
        self._kf.x[0] = z[0]
        self._kf.x[1] = z[1]

    @property
    def state(self) -> np.ndarray:
        """Full fused state ``[px, py, vx, vy, ax, ay]``."""
        return self._kf.x.copy()

    @property
    def position(self) -> np.ndarray:
        """Fused position ``[px, py]``."""
        return self._kf.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Fused velocity ``[vx, vy]``."""
        return self._kf.x[2:4].copy()

    @property
    def covariance(self) -> np.ndarray:
        """Full 6x6 state covariance."""
        return self._kf.P.copy()
