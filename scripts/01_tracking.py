"""칼만 추적: 잡음 위치 측정에서 궤적을 복원.

등속(constant-velocity) 모델 칼만 필터로 잡음 위치를 추적하고, 원측정·이동평균과
정확도를 비교한다.

    python scripts/01_tracking.py
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
from sensor_fusion.sim import noisy_position, true_trajectory  # noqa: E402


def rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def moving_average(z, w=5):
    k = np.ones(w) / w
    out = np.stack([np.convolve(z[:, i], k, mode="same") for i in range(z.shape[1])], axis=1)
    return out


def main() -> None:
    dt = 0.1
    n = 300
    t, pos, vel, acc = true_trajectory(n, dt=dt)
    meas = noisy_position(pos, sigma=2.0)

    # 상태 [px, py, vx, vy], 등속 모델
    F = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float
    )
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)  # 위치만 관측
    q = 10.0  # 프로세스 잡음 (기동표적 흡수; q 스윕으로 튜닝)
    Q = q * np.array(
        [[dt**3 / 3, 0, dt**2 / 2, 0],
         [0, dt**3 / 3, 0, dt**2 / 2],
         [dt**2 / 2, 0, dt, 0],
         [0, dt**2 / 2, 0, dt]], float
    )
    R = (2.0**2) * np.eye(2)

    kf = KalmanFilter(F, H, Q, R, x0=[meas[0, 0], meas[0, 1], 0, 0], P0=10 * np.eye(4))
    est = np.zeros((n, 2))
    vel_est = np.zeros((n, 2))
    for k in range(n):
        x = kf.step(meas[k])
        est[k] = x[:2]
        vel_est[k] = x[2:]

    ma = moving_average(meas, w=7)

    print("=== 위치 추정 RMSE (참값 대비) ===")
    print(f"raw measurement : {rmse(meas, pos):.3f} m")
    print(f"moving average  : {rmse(ma, pos):.3f} m  (위치만; 속도 추정 불가)")
    print(f"Kalman filter   : {rmse(est, pos):.3f} m  (위치 + 속도 동시 추정)")
    print(f"  velocity RMSE : {rmse(vel_est, vel):.3f} m/s (미분 잡음 없이 속도 획득)")
    print("\n주: 조밀한 위치측정만 있으면 이동평균도 경쟁력 있음. 칼만의 진짜 값어치는")
    print("   상태(속도) 추정과 다중 센서 융합 — 02_imu_fusion.py 참고.")

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(meas[:, 0], meas[:, 1], ".", ms=3, alpha=0.4, label="noisy measurement")
    ax.plot(pos[:, 0], pos[:, 1], "g-", lw=2, label="ground truth")
    ax.plot(est[:, 0], est[:, 1], "r-", lw=1.5, label="Kalman estimate")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("2D tracking: Kalman filter vs raw measurement")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(outdir / "01_tracking.png", dpi=130)
    print(f"\n[plot] {outdir / '01_tracking.png'}")


if __name__ == "__main__":
    main()
