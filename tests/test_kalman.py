"""칼만 필터 단위 테스트. 실행: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.kalman import KalmanFilter
from sensor_fusion.sim import noisy_position, true_trajectory


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_kalman_reduces_position_error():
    dt = 0.1
    n = 300
    _, pos, _, _ = true_trajectory(n, dt=dt)
    meas = noisy_position(pos, sigma=2.0)
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    q = 10.0
    Q = q * np.array(
        [[dt**3 / 3, 0, dt**2 / 2, 0], [0, dt**3 / 3, 0, dt**2 / 2],
         [dt**2 / 2, 0, dt, 0], [0, dt**2 / 2, 0, dt]], float
    )
    kf = KalmanFilter(F, H, Q, (2.0**2) * np.eye(2), [meas[0, 0], meas[0, 1], 0, 0], 10 * np.eye(4))
    est = np.array([kf.step(meas[k])[:2] for k in range(n)])
    # 필터가 원측정보다 참값에 가까워야 함
    assert _rmse(est, pos) < _rmse(meas, pos)


def test_covariance_symmetric_positive():
    F = np.eye(2)
    H = np.array([[1.0, 0.0]])
    kf = KalmanFilter(F, H, np.eye(2) * 0.1, np.array([[1.0]]), [0, 0], np.eye(2))
    for z in [1.0, 1.2, 0.9, 1.1]:
        kf.step([z])
    # 공분산은 대칭·양정치를 유지해야 함
    assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert np.all(np.linalg.eigvals(kf.P) > 0)


def test_multi_sensor_update_runs():
    F = np.eye(6)
    kf = KalmanFilter(F, np.zeros((2, 6)), np.eye(6) * 1e-3, np.eye(2), np.zeros(6), np.eye(6))
    H_imu = np.zeros((2, 6)); H_imu[0, 4] = H_imu[1, 5] = 1.0
    kf.predict()
    kf.update(np.array([0.1, 0.2]), H=H_imu, R=np.eye(2) * 0.25)
    assert np.all(np.isfinite(kf.x))
