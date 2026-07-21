"""비선형 운동 추적: 회전하는 표적(CTRV) — 선형 CV-KF vs EKF vs UKF.

CTRV(Constant Turn Rate & Velocity)는 등속으로 일정하게 선회하는 운동으로, 상태
[px,py,v,ψ,ψ']에 sin/cos가 들어가 강하게 비선형이다. 위치 측정(lidar류)은 선형이라
'운동 비선형'만 분리해 비교한다:
  - linear CV-KF : 등속 직선 가정 → 선회에서 구조적으로 뒤처짐
  - EKF          : CTRV를 야코비안으로 선형화
  - UKF          : 시그마 포인트로 비선형 전파

    python scripts/03_ctrv_ekf_ukf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.ekf import ExtendedKalmanFilter  # noqa: E402
from sensor_fusion.kalman import KalmanFilter  # noqa: E402
from sensor_fusion.ukf import UnscentedKalmanFilter  # noqa: E402

DT = 0.1
POS_STD = 2.0  # 위치 측정 잡음[m]
EPS = 1e-4


def ctrv_f(x, dt):
    px, py, v, psi, w = x
    if abs(w) > EPS:
        npx = px + v / w * (np.sin(psi + w * dt) - np.sin(psi))
        npy = py + v / w * (-np.cos(psi + w * dt) + np.cos(psi))
    else:
        npx = px + v * np.cos(psi) * dt
        npy = py + v * np.sin(psi) * dt
    return np.array([npx, npy, v, psi + w * dt, w])


def ctrv_F(x, dt):
    px, py, v, psi, w = x
    J = np.eye(5)
    s0, c0 = np.sin(psi), np.cos(psi)
    if abs(w) > EPS:
        s1, c1 = np.sin(psi + w * dt), np.cos(psi + w * dt)
        J[0, 2] = (s1 - s0) / w
        J[0, 3] = v / w * (c1 - c0)
        J[0, 4] = v * (dt * c1 * w - (s1 - s0)) / w**2
        J[1, 2] = (c0 - c1) / w
        J[1, 3] = v / w * (s1 - s0)
        J[1, 4] = v * (dt * s1 * w - (c0 - c1)) / w**2
    else:
        J[0, 2] = c0 * dt; J[0, 3] = -v * s0 * dt
        J[1, 2] = s0 * dt; J[1, 3] = v * c0 * dt
    J[3, 4] = dt
    return J


def main() -> None:
    rng = np.random.default_rng(1)
    N = 220

    # 참 궤적: 구간마다 회전율을 바꿔 곡선 기동
    x = np.array([0.0, 0.0, 12.0, 0.0, 0.0])
    truth, meas = [], []
    for k in range(N):
        w = 0.0 if k < 30 else (0.5 if k < 100 else (-0.45 if k < 170 else 0.3))
        x = ctrv_f(np.array([x[0], x[1], x[2], x[3], w]), DT)
        truth.append(x.copy())
        meas.append([x[0] + rng.normal(0, POS_STD), x[1] + rng.normal(0, POS_STD)])
    truth = np.array(truth); meas = np.array(meas)

    Hpos = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], float)
    R = np.diag([POS_STD**2, POS_STD**2])
    Q = np.diag([0.1, 0.1, 1.0, 0.05, 0.3])  # CTRV 프로세스 잡음
    x0 = np.array([meas[0, 0], meas[0, 1], 8.0, 0.0, 0.0])
    P0 = np.diag([4, 4, 25, 1.0, 1.0]).astype(float)

    # 1) 선형 CV-KF (등속 가정)
    Fcv = np.array([[1,0,DT,0],[0,1,0,DT],[0,0,1,0],[0,0,0,1]], float)
    Hcv = np.array([[1,0,0,0],[0,1,0,0]], float)
    Qcv = 5.0 * np.array([[DT**3/3,0,DT**2/2,0],[0,DT**3/3,0,DT**2/2],
                          [DT**2/2,0,DT,0],[0,DT**2/2,0,DT]], float)
    kf = KalmanFilter(Fcv, Hcv, Qcv, R, [meas[0,0],meas[0,1],12,0], np.diag([4,4,50,50]).astype(float))
    est_cv = np.array([(kf.predict(), kf.update(z), kf.x[:2].copy())[-1] for z in meas])

    # 2) EKF (CTRV)
    ekf = ExtendedKalmanFilter(ctrv_f, ctrv_F, lambda s: Hpos @ s, lambda s: Hpos, Q, R, x0, P0.copy())
    est_ekf = np.array([ekf.step(z, DT)[:2] for z in meas])

    # 3) UKF (CTRV)
    ukf = UnscentedKalmanFilter(5, 2, ctrv_f, lambda s: Hpos @ s, Q, R, x0, P0.copy())
    est_ukf = np.array([ukf.step(z, DT)[:2] for z in meas])

    def rmse(e):
        return float(np.sqrt(np.mean(np.sum((e - truth[:, :2])**2, axis=1))))

    print("=== 회전 표적(CTRV) 추적 위치 RMSE ===")
    print(f"raw measurement       : {rmse(meas):.3f} m")
    print(f"linear CV-KF          : {rmse(est_cv):.3f} m")
    print(f"EKF (CTRV)            : {rmse(est_ekf):.3f} m")
    print(f"UKF (CTRV)            : {rmse(est_ukf):.3f} m")

    # 회전 구간만
    turning = np.array([k for k in range(N) if 30 <= k < 170])
    def rmse_sub(e, idx):
        return float(np.sqrt(np.mean(np.sum((e[idx] - truth[idx, :2])**2, axis=1))))
    print(f"\n선회 구간만:  CV-KF={rmse_sub(est_cv,turning):.3f}  "
          f"EKF={rmse_sub(est_ekf,turning):.3f}  UKF={rmse_sub(est_ukf,turning):.3f} m")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(meas[:,0], meas[:,1], ".", ms=3, alpha=0.3, label="measurement")
    ax1.plot(truth[:,0], truth[:,1], "g-", lw=2, label="ground truth")
    ax1.plot(est_cv[:,0], est_cv[:,1], lw=1, alpha=0.8, label="linear CV-KF")
    ax1.plot(est_ekf[:,0], est_ekf[:,1], lw=1, label="EKF")
    ax1.plot(est_ukf[:,0], est_ukf[:,1], lw=1.4, label="UKF")
    ax1.set_aspect("equal"); ax1.legend(); ax1.set_title("Turning target (CTRV): CV-KF vs EKF vs UKF")
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]")
    t = np.arange(N) * DT
    ax2.plot(t, np.hypot(*(est_cv-truth[:,:2]).T), alpha=0.8, label="linear CV-KF")
    ax2.plot(t, np.hypot(*(est_ekf-truth[:,:2]).T), label="EKF")
    ax2.plot(t, np.hypot(*(est_ukf-truth[:,:2]).T), lw=1.4, label="UKF")
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("position error [m]"); ax2.set_title("Error vs time")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path("outputs") / "03_ctrv_ekf_ukf.png", dpi=130)
    print("\n[plot] outputs/03_ctrv_ekf_ukf.png")


if __name__ == "__main__":
    main()
