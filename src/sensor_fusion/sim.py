"""2D 이동체 궤적 시뮬레이션 + 잡음 센서 모델.

로봇/드론이 곡선 궤적을 그리며 움직이고, 두 종류의 잡음 센서가 이를 관측한다:
- 위치 센서(GPS류): 위치를 직접 측정하나 잡음이 크고 갱신이 느릴 수 있음
- 가속도 센서(IMU류): 가속도를 측정하나 적분하면 드리프트가 쌓임
"""

from __future__ import annotations

import numpy as np


def true_trajectory(n: int, dt: float = 0.1, seed: int = 0):
    """부드럽게 휘는 2D 궤적의 참값(위치·속도·가속도)을 만든다."""
    t = np.arange(n) * dt
    # 8자 비슷한 궤적
    px = 20.0 * np.sin(0.3 * t)
    py = 12.0 * np.sin(0.6 * t)
    vx = np.gradient(px, dt)
    vy = np.gradient(py, dt)
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    pos = np.stack([px, py], axis=1)
    vel = np.stack([vx, vy], axis=1)
    acc = np.stack([ax, ay], axis=1)
    return t, pos, vel, acc


def noisy_position(pos: np.ndarray, sigma: float = 2.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return pos + rng.normal(0, sigma, pos.shape)


def noisy_accel(
    acc: np.ndarray, sigma: float = 0.5, bias: float = 0.1, seed: int = 2
) -> np.ndarray:
    """IMU류 가속도 측정: 백색잡음 + 고정 바이어스(드리프트 원인)."""
    rng = np.random.default_rng(seed)
    return acc + rng.normal(0, sigma, acc.shape) + bias
