"""EKF/UKF 단위 테스트. 실행: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.ekf import ExtendedKalmanFilter
from sensor_fusion.ukf import UnscentedKalmanFilter

DT = 0.1


def _linear_setup():
    F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    Q = np.eye(4) * 0.01
    R = np.eye(2) * 1.0
    rng = np.random.default_rng(0)
    x = np.array([0.0, 0.0, 2.0, 1.0])
    truth, meas = [], []
    for _ in range(100):
        x = F @ x
        truth.append(x[:2].copy())
        meas.append(x[:2] + rng.normal(0, 1.0, 2))
    return F, H, Q, R, np.array(truth), np.array(meas)


def test_ekf_reduces_error_linear():
    F, H, Q, R, truth, meas = _linear_setup()
    ekf = ExtendedKalmanFilter(lambda s, dt: F @ s, lambda s, dt: F,
                               lambda s: H @ s, lambda s: H, Q, R,
                               [meas[0, 0], meas[0, 1], 0, 0], np.eye(4) * 10)
    est = np.array([ekf.step(z, DT)[:2] for z in meas])
    err_est = np.sqrt(np.mean(np.sum((est - truth) ** 2, 1)))
    err_raw = np.sqrt(np.mean(np.sum((meas - truth) ** 2, 1)))
    assert err_est < err_raw


def test_ukf_reduces_error_linear():
    F, H, Q, R, truth, meas = _linear_setup()
    ukf = UnscentedKalmanFilter(4, 2, lambda s, dt: F @ s, lambda s: H @ s, Q, R,
                                [meas[0, 0], meas[0, 1], 0, 0], np.eye(4) * 10)
    est = np.array([ukf.step(z, DT)[:2] for z in meas])
    err_est = np.sqrt(np.mean(np.sum((est - truth) ** 2, 1)))
    err_raw = np.sqrt(np.mean(np.sum((meas - truth) ** 2, 1)))
    assert err_est < err_raw


def test_ekf_ukf_agree_on_linear():
    # 선형 모델에서 EKF와 UKF 추정은 거의 같아야 한다
    F, H, Q, R, truth, meas = _linear_setup()
    x0 = [meas[0, 0], meas[0, 1], 0, 0]
    ekf = ExtendedKalmanFilter(lambda s, dt: F @ s, lambda s, dt: F,
                               lambda s: H @ s, lambda s: H, Q, R, x0, np.eye(4) * 10)
    ukf = UnscentedKalmanFilter(4, 2, lambda s, dt: F @ s, lambda s: H @ s, Q, R, x0, np.eye(4) * 10)
    e = np.array([ekf.step(z, DT)[:2] for z in meas])
    u = np.array([ukf.step(z, DT)[:2] for z in meas])
    assert np.max(np.abs(e - u)) < 0.5


def test_ukf_covariance_symmetric():
    F, H, Q, R, truth, meas = _linear_setup()
    ukf = UnscentedKalmanFilter(4, 2, lambda s, dt: F @ s, lambda s: H @ s, Q, R,
                                [0, 0, 0, 0], np.eye(4) * 10)
    for z in meas[:20]:
        ukf.step(z, DT)
    assert np.allclose(ukf.P, ukf.P.T, atol=1e-6)
