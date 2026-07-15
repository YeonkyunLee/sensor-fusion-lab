"""범용 선형 칼만 필터.

로봇 상태추정의 기본 도구. DSP의 관점에서 칼만 필터는 '시변 최적 IIR 필터'로,
측정 잡음과 프로세스 잡음의 비율에 따라 대역폭을 스스로 조절하는 추정기다.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """이산시간 선형 칼만 필터.

    상태:   x_k = F x_{k-1} + w,   w ~ N(0, Q)
    측정:   z_k = H x_k + v,       v ~ N(0, R)
    """

    def __init__(
        self,
        F: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
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

    def update(
        self, z: np.ndarray, H: np.ndarray | None = None, R: np.ndarray | None = None
    ) -> None:
        """측정 갱신. H/R을 주면 그 센서 모델로, 없으면 기본값 사용(다중 센서 융합용)."""
        H = self.H if H is None else np.asarray(H, float)
        R = self.R if R is None else np.asarray(R, float)
        z = np.asarray(z, float).reshape(-1)
        y = z - H @ self.x  # 잔차(innovation)
        S = H @ self.P @ H.T + R  # 잔차 공분산
        K = self.P @ H.T @ np.linalg.inv(S)  # 칼만 이득
        self.x = self.x + K @ y
        # Joseph 형태: 수치적으로 공분산 대칭·양정치 유지에 유리
        A = self._I - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T

    def step(self, z: np.ndarray | None) -> np.ndarray:
        """predict 후, 측정이 있으면 update. 추정 상태를 반환."""
        self.predict()
        if z is not None:
            self.update(z)
        return self.x.copy()
