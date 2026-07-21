"""확장 칼만 필터 (EKF).

비선형 운동/측정 모델을 각 시점에서 1차 테일러 전개(야코비안)로 선형화해 칼만
갱신을 적용한다. 강한 비선형·큰 불확실도에서는 선형화 오차가 커진다(→ UKF 비교).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class ExtendedKalmanFilter:
    """일반 EKF.

    f(x, dt)   : 비선형 상태천이
    F_jac(x,dt): f의 야코비안 (∂f/∂x)
    h(x)       : 비선형 측정함수
    H_jac(x)   : h의 야코비안 (∂h/∂x)
    residual_z : 측정 잔차 보정(각도 wrap 등). 기본은 단순 차.
    """

    def __init__(
        self,
        f: Callable, F_jac: Callable, h: Callable, H_jac: Callable,
        Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray,
        residual_z: Callable | None = None,
    ):
        self.f, self.F_jac, self.h, self.H_jac = f, F_jac, h, H_jac
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.x = np.asarray(x0, float).reshape(-1)
        self.P = np.asarray(P0, float)
        self._I = np.eye(self.P.shape[0])
        self._res = residual_z or (lambda a, b: a - b)

    def predict(self, dt: float = 1.0) -> None:
        Fj = self.F_jac(self.x, dt)
        self.x = self.f(self.x, dt)
        self.P = Fj @ self.P @ Fj.T + self.Q

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, float).reshape(-1)
        Hj = self.H_jac(self.x)
        y = self._res(z, self.h(self.x))
        S = Hj @ self.P @ Hj.T + self.R
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        A = self._I - K @ Hj
        self.P = A @ self.P @ A.T + K @ self.R @ K.T  # Joseph 형태

    def step(self, z: np.ndarray | None, dt: float = 1.0) -> np.ndarray:
        self.predict(dt)
        if z is not None:
            self.update(z)
        return self.x.copy()
