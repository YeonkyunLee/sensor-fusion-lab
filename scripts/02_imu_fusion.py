"""다중 센서 융합: 위치 센서 + IMU(가속도).

상태 [px,py,vx,vy,ax,ay]의 등가속 모델을 두 센서로 갱신한다.
- 위치 센서(GPS류): 위치를 관측, 잡음 큼, 중간에 끊김(outage)
- IMU: 가속도를 관측, 고속, 그러나 단독 적분하면 드리프트

칼만이 둘을 융합하면, 위치센서 끊김 구간을 IMU로 메우고(dead-reckoning) 위치센서
복귀 시 드리프트를 교정한다. 어떤 단일 센서보다 나은 추정 — 칼만의 핵심 가치.

    python scripts/02_imu_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.kalman import KalmanFilter  # noqa: E402
from sensor_fusion.sim import noisy_accel, noisy_position, true_trajectory  # noqa: E402


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> None:
    dt = 0.1
    n = 300
    t, pos, vel, acc = true_trajectory(n, dt=dt)
    z_pos = noisy_position(pos, sigma=2.0)
    z_acc = noisy_accel(acc, sigma=0.5, bias=0.0)  # 바이어스 0(단순화)

    # 위치 센서 끊김: k=120~180 (6초)
    outage = set(range(120, 180))

    # 등가속 모델: 상태 [px,py,vx,vy,ax,ay]
    F = np.eye(6)
    F[0, 2] = F[1, 3] = dt
    F[2, 4] = F[3, 5] = dt
    F[0, 4] = F[1, 5] = 0.5 * dt**2
    Q = np.eye(6) * 1e-3
    Q[4, 4] = Q[5, 5] = 0.5  # 가속도 변화(저크)를 프로세스 잡음으로

    H_pos = np.zeros((2, 6)); H_pos[0, 0] = H_pos[1, 1] = 1.0
    H_imu = np.zeros((2, 6)); H_imu[0, 4] = H_imu[1, 5] = 1.0
    R_pos = (2.0**2) * np.eye(2)
    R_imu = (0.5**2) * np.eye(2)

    x0 = [z_pos[0, 0], z_pos[0, 1], 0, 0, 0, 0]
    kf = KalmanFilter(F, H_pos, Q, R_pos, x0=x0, P0=np.eye(6) * 10)

    fused = np.zeros((n, 2))
    for k in range(n):
        kf.predict()
        kf.update(z_acc[k], H=H_imu, R=R_imu)  # IMU는 항상 갱신
        if k not in outage:  # 위치센서는 가용할 때만
            kf.update(z_pos[k], H=H_pos, R=R_pos)
        fused[k] = kf.x[:2]

    # 비교: IMU 단독 dead-reckoning (초기 위치/속도에서 가속도 이중적분)
    dr = np.zeros((n, 2))
    v = np.array([0.0, 0.0]); p = np.array(pos[0], float)
    for k in range(n):
        p = p + v * dt + 0.5 * z_acc[k] * dt**2
        v = v + z_acc[k] * dt
        dr[k] = p

    out_idx = sorted(outage)
    print("=== 다중 센서 융합 RMSE (참값 대비) ===")
    print(f"위치센서 원측정      : {rmse(z_pos, pos):.3f} m")
    print(f"IMU 단독 dead-reckon : {rmse(dr, pos):.3f} m  (적분 드리프트)")
    print(f"칼만 융합            : {rmse(fused, pos):.3f} m")
    print(f"\n위치센서 끊김(k=120~180) 구간만:")
    print(f"  IMU 단독   @outage : {rmse(dr[out_idx], pos[out_idx]):.3f} m")
    print(f"  칼만 융합  @outage : {rmse(fused[out_idx], pos[out_idx]):.3f} m  ← IMU로 관성주행")

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(z_pos[:, 0], z_pos[:, 1], ".", ms=3, alpha=0.3, label="position sensor")
    ax.plot(pos[:, 0], pos[:, 1], "g-", lw=2, label="ground truth")
    ax.plot(fused[:, 0], fused[:, 1], "r-", lw=1.5, label="Kalman fusion")
    ax.plot(pos[out_idx, 0], pos[out_idx, 1], "b-", lw=3, alpha=0.4, label="position outage")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Sensor fusion: position + IMU (with position outage)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(outdir / "02_imu_fusion.png", dpi=130)
    print(f"\n[plot] {outdir / '02_imu_fusion.png'}")


if __name__ == "__main__":
    main()
