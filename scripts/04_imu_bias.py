"""IMU 바이어스 온라인 추정 (상태 증강).

IMU 가속도계는 느리게 변하는 바이어스를 갖는다. 추정하지 않으면 이중적분으로 위치가
드리프트한다. 상태에 바이어스를 넣어(증강) 위치 측정으로 온라인 관측하면 교정된다.

두 필터를 비교:
  - no-bias  : 상태 [p,v], 바이어스=0 가정 → 잔여 드리프트
  - bias-aug : 상태 [p,v,b], 바이어스를 함께 추정 → 교정

    python scripts/04_imu_bias.py
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
from sensor_fusion.sim import true_trajectory  # noqa: E402

DT = 0.1
POS_STD = 1.5
ACC_STD = 0.3
TRUE_BIAS = np.array([0.25, -0.18])  # 미지의 가속도계 바이어스 [m/s²]


def build_no_bias():
    # 상태 [px,py,vx,vy], 제어입력=측정가속도(바이어스 미보정)
    F = np.eye(4); F[0, 2] = F[1, 3] = DT
    B = np.array([[0.5*DT**2, 0], [0, 0.5*DT**2], [DT, 0], [0, DT]])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
    Q = np.diag([0.05, 0.05, 0.1, 0.1])
    R = np.diag([POS_STD**2, POS_STD**2])
    return KalmanFilter(F, H, Q, R, [0, 0, 0, 0], np.diag([4, 4, 4, 4]).astype(float), B=B)


def build_bias_aug():
    # 상태 [px,py,vx,vy,bax,bay]; 참가속 = 측정 - 바이어스
    F = np.eye(6)
    F[0, 2] = F[1, 3] = DT
    F[0, 4] = F[1, 5] = -0.5*DT**2
    F[2, 4] = F[3, 5] = -DT
    B = np.array([[0.5*DT**2, 0], [0, 0.5*DT**2], [DT, 0], [0, DT], [0, 0], [0, 0]])
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], float)
    Q = np.diag([0.05, 0.05, 0.1, 0.1, 1e-4, 1e-4])  # 바이어스=random walk(작은 Q)
    R = np.diag([POS_STD**2, POS_STD**2])
    return KalmanFilter(F, H, Q, R, [0, 0, 0, 0, 0, 0], np.diag([4, 4, 4, 4, 1, 1]).astype(float), B=B)


def main() -> None:
    rng = np.random.default_rng(2)
    t, pos, vel, acc = true_trajectory(300, dt=DT)
    N = len(t)

    # IMU 측정 = 참가속 + 바이어스 + 잡음
    imu = acc + TRUE_BIAS + rng.normal(0, ACC_STD, acc.shape)
    # 위치 측정(잡음), 중간 구간(k=120~200) 끊김
    zpos = pos + rng.normal(0, POS_STD, pos.shape)
    outage = set(range(120, 200))

    kf0 = build_no_bias()
    kf1 = build_bias_aug()
    est0 = np.zeros((N, 2)); est1 = np.zeros((N, 2)); bias_est = np.zeros((N, 2))
    for k in range(N):
        z = None if k in outage else zpos[k]
        kf0.step(z, u=imu[k]); est0[k] = kf0.x[:2]
        kf1.step(z, u=imu[k]); est1[k] = kf1.x[:2]; bias_est[k] = kf1.x[4:6]

    def rmse(e):
        return float(np.sqrt(np.mean(np.sum((e - pos)**2, axis=1))))

    oi = sorted(outage)
    print("=== IMU 바이어스 추정: 위치 RMSE ===")
    print(f"no-bias  : {rmse(est0):.3f} m")
    print(f"bias-aug : {rmse(est1):.3f} m")
    print(f"\n끊김 구간(k=120~200)만:")
    print(f"  no-bias ={np.sqrt(np.mean(np.sum((est0[oi]-pos[oi])**2,1))):.3f}  "
          f"bias-aug={np.sqrt(np.mean(np.sum((est1[oi]-pos[oi])**2,1))):.3f} m")
    print(f"\n추정 바이어스(최종) = [{bias_est[-1,0]:.3f}, {bias_est[-1,1]:.3f}]  "
          f"참값 = [{TRUE_BIAS[0]:.3f}, {TRUE_BIAS[1]:.3f}]")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    ax1.plot(pos[:,0], pos[:,1], "g-", lw=2, label="ground truth")
    ax1.plot(est0[:,0], est0[:,1], lw=1, alpha=0.8, label="no-bias")
    ax1.plot(est1[:,0], est1[:,1], lw=1.2, label="bias-aug")
    ax1.plot(pos[oi,0], pos[oi,1], "b-", lw=3, alpha=0.3, label="outage")
    ax1.set_aspect("equal"); ax1.legend(fontsize=8); ax1.set_title("Trajectory"); ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")

    ax2.axhline(TRUE_BIAS[0], color="C0", ls="--", lw=0.8, label="true bax")
    ax2.axhline(TRUE_BIAS[1], color="C1", ls="--", lw=0.8, label="true bay")
    ax2.plot(t, bias_est[:,0], "C0-", label="est bax")
    ax2.plot(t, bias_est[:,1], "C1-", label="est bay")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("accel bias [m/s²]"); ax2.set_title("Bias convergence"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    ax3.plot(t, np.hypot(*(est0-pos).T), alpha=0.8, label="no-bias")
    ax3.plot(t, np.hypot(*(est1-pos).T), lw=1.2, label="bias-aug")
    ax3.axvspan(120*DT, 200*DT, color="b", alpha=0.1, label="outage")
    ax3.set_xlabel("time [s]"); ax3.set_ylabel("position error [m]"); ax3.set_title("Error vs time"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    fig.suptitle("Online IMU bias estimation via state augmentation")
    fig.tight_layout()
    fig.savefig(Path("outputs") / "04_imu_bias.png", dpi=130)
    print("\n[plot] outputs/04_imu_bias.png")


if __name__ == "__main__":
    main()
