"""무향 칼만 필터 (UKF, van der Merwe 스케일드 시그마포인트).

야코비안 대신 시그마 포인트를 비선형 함수에 직접 통과시켜 평균·공분산을 추정한다.
2차까지 정확해 EKF의 선형화 오차를 줄인다. 각도 등 순환량은 residual/mean 훅으로 처리.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class UnscentedKalmanFilter:
    def __init__(
        self,
        dim_x: int, dim_z: int, f: Callable, h: Callable,
        Q: np.ndarray, R: np.ndarray, x0: np.ndarray, P0: np.ndarray,
        alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0,
        residual_z: Callable | None = None, mean_z: Callable | None = None,
    ):
        self.n, self.dz = dim_x, dim_z
        self.f, self.h = f, h
        self.Q, self.R = np.asarray(Q, float), np.asarray(R, float)
        self.x = np.asarray(x0, float).reshape(-1)
        self.P = np.asarray(P0, float)
        self.lmbda = alpha**2 * (self.n + kappa) - self.n
        self._res_z = residual_z or (lambda a, b: a - b)
        self._mean_z = mean_z or (lambda sig, wm: wm @ sig)

        # 가중치
        c = self.n + self.lmbda
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lmbda / c
        self.Wc[0] = self.lmbda / c + (1 - alpha**2 + beta)

    def _sigma(self, x, P):
        c = self.n + self.lmbda
        U = np.linalg.cholesky(c * (P + 1e-9 * np.eye(self.n)))
        pts = np.zeros((2 * self.n + 1, self.n))
        pts[0] = x
        for i in range(self.n):
            pts[i + 1] = x + U[:, i]
            pts[self.n + i + 1] = x - U[:, i]
        return pts

    def predict(self, dt: float = 1.0) -> None:
        pts = self._sigma(self.x, self.P)
        self._fp = np.array([self.f(p, dt) for p in pts])  # 전개된 시그마
        self.x = self.Wm @ self._fp
        P = self.Q.copy()
        for i in range(2 * self.n + 1):
            d = self._fp[i] - self.x
            P += self.Wc[i] * np.outer(d, d)
        self.P = P

    def update(self, z: np.ndarray) -> None:
        z = np.asarray(z, float).reshape(-1)
        zp = np.array([self.h(p) for p in self._fp])  # 측정공간 시그마
        zbar = self._mean_z(zp, self.Wm)
        Pzz = self.R.copy()
        Pxz = np.zeros((self.n, self.dz))
        for i in range(2 * self.n + 1):
            dz = self._res_z(zp[i], zbar)
            dx = self._fp[i] - self.x
            Pzz += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)
        K = Pxz @ np.linalg.inv(Pzz)
        self.x = self.x + K @ self._res_z(z, zbar)
        self.P = self.P - K @ Pzz @ K.T

    def step(self, z: np.ndarray | None, dt: float = 1.0) -> np.ndarray:
        self.predict(dt)
        if z is not None:
            self.update(z)
        return self.x.copy()
