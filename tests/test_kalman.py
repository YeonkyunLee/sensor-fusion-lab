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


def test_control_input_prediction():
    # 제어입력 u가 예측에 반영되는지: 등속 모델에 가속 입력
    dt = 0.1
    F = np.array([[1, dt], [0, 1]], float)  # [pos, vel]
    B = np.array([[0.5 * dt**2], [dt]])
    H = np.array([[1.0, 0.0]])
    kf = KalmanFilter(F, H, np.eye(2) * 1e-6, np.array([[1.0]]), [0.0, 0.0], np.eye(2), B=B)
    for _ in range(10):
        kf.predict(u=[1.0])  # 가속 1 m/s²
    # 1초 등가속: pos ≈ 0.5*a*t² = 0.5, vel ≈ 1.0
    assert abs(kf.x[1] - 1.0) < 1e-6
    assert 0.4 < kf.x[0] < 0.6


def test_bias_augmentation_helps():
    # 바이어스 증강 필터가 끊김 구간에서 no-bias보다 위치 RMSE가 낮아야 함
    # 간단 시뮬: 1D 등속 + 가속 바이어스, 위치 측정
    rng = np.random.default_rng(0)
    dt, n, bias = 0.1, 200, 0.3
    pos, vel = 0.0, 1.0
    truth, zpos, imu = [], [], []
    for _ in range(n):
        a_true = 0.0
        pos += vel * dt + 0.5 * a_true * dt**2
        vel += a_true * dt
        truth.append(pos)
        zpos.append(pos + rng.normal(0, 1.0))
        imu.append(a_true + bias + rng.normal(0, 0.2))
    truth = np.array(truth)

    # no-bias: [p,v]
    F0 = np.array([[1, dt], [0, 1]], float); B0 = np.array([[0.5*dt**2], [dt]])
    kf0 = KalmanFilter(F0, np.array([[1.0, 0]]), np.eye(2)*0.05, np.array([[1.0]]), [0, 1], np.eye(2)*4, B=B0)
    # bias-aug: [p,v,b]
    F1 = np.array([[1, dt, -0.5*dt**2], [0, 1, -dt], [0, 0, 1]], float)
    B1 = np.array([[0.5*dt**2], [dt], [0]])
    kf1 = KalmanFilter(F1, np.array([[1.0, 0, 0]]), np.diag([0.05, 0.05, 1e-4]), np.array([[1.0]]), [0, 1, 0], np.diag([4, 4, 1.0]), B=B1)
    outage = set(range(140, 190))  # 측정 끊김 → dead-reckoning
    e0, e1 = [], []
    for k in range(n):
        z = None if k in outage else [zpos[k]]
        kf0.step(z, u=[imu[k]]); e0.append(kf0.x[0])
        kf1.step(z, u=[imu[k]]); e1.append(kf1.x[0])
    oi = sorted(outage)
    # 끊김 구간에서 바이어스 보정 dead-reckoning이 더 정확해야 함
    rmse0 = np.sqrt(np.mean((np.array(e0)[oi] - truth[oi]) ** 2))
    rmse1 = np.sqrt(np.mean((np.array(e1)[oi] - truth[oi]) ** 2))
    assert rmse1 < rmse0
    # 바이어스도 대략 수렴
    assert abs(kf1.x[2] - bias) < 0.2


def test_multi_sensor_update_runs():
    F = np.eye(6)
    kf = KalmanFilter(F, np.zeros((2, 6)), np.eye(6) * 1e-3, np.eye(2), np.zeros(6), np.eye(6))
    H_imu = np.zeros((2, 6)); H_imu[0, 4] = H_imu[1, 5] = 1.0
    kf.predict()
    kf.update(np.array([0.1, 0.2]), H=H_imu, R=np.eye(2) * 0.25)
    assert np.all(np.isfinite(kf.x))
