"""2D 시각-관성 오도메트리 (VIO): IMU 고속 예측 + 저속 시각 방위 보정.

모노큘러 카메라는 거리를 못 재고 랜드마크 '방위(bearing)'만 준다. IMU(가속도+자이로)는
고속으로 pose를 예측하지만 이중적분 드리프트가 쌓인다. 둘을 EKF로 타이트하게 융합하면
드리프트가 억제된다 — 이것이 VIO의 핵심.

    python scripts/08_vio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.05                     # 20Hz IMU
VIS_EVERY = 6                 # 시각 갱신은 저속(~3.3Hz)
ACC_STD, GYR_STD = 0.25, 0.02
B_STD = np.radians(1.0)      # 시각 방위 측정 잡음


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    rng = np.random.default_rng(0)

    # 알려진 랜드마크(시각 특징점)
    landmarks = np.array([[10, 6], [22, -4], [4, -8], [26, 10], [15, 14],
                          [30, 2], [0, 5], [18, -12]], float)

    # 참 궤적 생성(곡선 주행) + 참 IMU(월드 가속/각속도)
    n = 500
    true = np.zeros((n, 5))     # px,py,theta,vx,vy
    x = np.array([0, 0, 0, 6.0, 0.0])
    accs, gyrs = [], []
    for k in range(n):
        w = 0.0 if k < 40 else (0.4 if k < 160 else (-0.35 if k < 320 else 0.3))
        th = x[2]
        # 목표: 등속 곡선 → 구심 가속(월드). 간단히 body accel=0, 회전만.
        aw = np.array([0.0, 0.0])
        # 참 상태 전개
        x = np.array([x[0] + x[3]*DT, x[1] + x[4]*DT, wrap(th + w*DT),
                      6.0*np.cos(wrap(th + w*DT)), 6.0*np.sin(wrap(th + w*DT))])
        true[k] = x
        # body-frame 가속(다음 속도-현재 속도)/dt를 body로 회전
        accs.append(aw); gyrs.append(w)
    # IMU 측정 = 참 + 바이어스/잡음
    gyr_meas = np.array(gyrs) + rng.normal(0, GYR_STD, n) + 0.01   # 자이로 바이어스
    # 참 body accel 재계산: a_world = dv/dt, a_body = R^T a_world
    vel = true[:, 3:5]
    a_world = np.vstack([[0, 0], np.diff(vel, axis=0) / DT])
    acc_meas = np.zeros((n, 2))
    for k in range(n):
        c, s = np.cos(true[k, 2]), np.sin(true[k, 2])
        Rt = np.array([[c, s], [-s, c]])
        acc_meas[k] = Rt @ a_world[k] + rng.normal(0, ACC_STD, 2)

    def predict(x, P, ab, w):
        th = x[2]; c, s = np.cos(th), np.sin(th)
        aw = np.array([c*ab[0] - s*ab[1], s*ab[0] + c*ab[1]])
        xn = np.array([x[0] + x[3]*DT + 0.5*aw[0]*DT**2,
                       x[1] + x[4]*DT + 0.5*aw[1]*DT**2,
                       wrap(th + w*DT), x[3] + aw[0]*DT, x[4] + aw[1]*DT])
        F = np.eye(5)
        F[0, 3] = DT; F[1, 4] = DT
        F[0, 2] = 0.5*DT**2*(-aw[1]); F[1, 2] = 0.5*DT**2*aw[0]
        F[3, 2] = DT*(-aw[1]); F[4, 2] = DT*aw[0]
        Q = np.diag([1e-4, 1e-4, (GYR_STD*DT)**2, (ACC_STD*DT)**2, (ACC_STD*DT)**2]) * 4
        return xn, F @ P @ F.T + Q

    def run(use_vision):
        x = np.array([0, 0, 0, 6.0, 0.0], float)
        P = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])
        est = np.zeros((n, 2))
        for k in range(n):
            x, P = predict(x, P, acc_meas[k], gyr_meas[k])
            if use_vision and k % VIS_EVERY == 0:
                for lm in landmarks:
                    dx, dy = lm[0]-x[0], lm[1]-x[1]; q = dx*dx+dy*dy
                    if q < 1.0 or q > 35**2:
                        continue
                    zhat = wrap(np.arctan2(dy, dx) - x[2])
                    z = wrap(np.arctan2(lm[1]-true[k, 1], lm[0]-true[k, 0]) - true[k, 2]
                             + rng.normal(0, B_STD))
                    H = np.zeros((1, 5)); H[0, 0] = dy/q; H[0, 1] = -dx/q; H[0, 2] = -1
                    S = float((H @ P @ H.T)[0, 0]) + B_STD**2
                    K = (P @ H.T / S).ravel()
                    x = x + K * wrap(z - zhat); x[2] = wrap(x[2])
                    P = (np.eye(5) - np.outer(K, H)) @ P
            est[k] = x[:2]
        return est

    imu_only = run(use_vision=False)
    vio = run(use_vision=True)

    def rmse(e):
        return float(np.sqrt(np.mean(np.sum((e - true[:, :2])**2, axis=1))))
    print("=== 2D VIO: IMU 단독 vs 시각-관성 융합 ===")
    print(f"IMU 단독(dead-reckoning): {rmse(imu_only):.2f} m")
    print(f"VIO(IMU + 시각 방위)    : {rmse(vio):.2f} m")
    print(f"→ 시각 융합이 드리프트를 {rmse(imu_only)/max(rmse(vio),1e-6):.0f}배 줄임")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true")
    ax.plot(imu_only[:, 0], imu_only[:, 1], "r--", lw=1.2, label=f"IMU only ({rmse(imu_only):.1f}m)")
    ax.plot(vio[:, 0], vio[:, 1], "b-", lw=1.4, label=f"VIO ({rmse(vio):.2f}m)")
    ax.plot(landmarks[:, 0], landmarks[:, 1], "k*", ms=11, label="visual features")
    ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Visual-Inertial Odometry: IMU + monocular bearing fusion")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(Path("outputs") / "08_vio.png", dpi=130)
    print("\n[plot] outputs/08_vio.png")
    return rmse(imu_only), rmse(vio)


if __name__ == "__main__":
    main()
