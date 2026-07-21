"""EKF-SLAM: 로봇 자기위치추정 + 지도작성을 동시에.

로봇이 2D를 주행하며 range-bearing 센서로 랜드마크를 관측한다. 상태를
[로봇 pose(x,y,θ), 랜드마크1(x,y), 랜드마크2(x,y), ...]로 확장하고, 관측할 때마다
로봇 pose와 지도를 함께 갱신한다. 데이터 연관(어느 랜드마크인지)은 알려진 것으로 가정.

    python scripts/05_ekf_slam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DT = 0.1
SENSOR_RANGE = 35.0               # 넓게 → 항상 여러 랜드마크(정립된 것 포함) 관측 → 헤딩 관측성
R_STD, B_STD = 0.5, 0.02          # 관측 잡음 (range[m], bearing[rad])
V_STD, W_STD = 0.10, 0.03         # 제어(오도메트리) 잡음


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def motion(pose, v, w):
    x, y, th = pose
    return np.array([x + v * DT * np.cos(th), y + v * DT * np.sin(th), wrap(th + w * DT)])


def main(plot: bool = True):
    rng = np.random.default_rng(0)

    # 진짜 랜드마크 지도 — 궤적 전체에 분포(로봇이 늘 일부를 관측)
    landmarks = np.array([
        [12, 8], [25, -6], [8, -12], [30, 16], [20, 22],
        [42, 30], [52, 46], [33, 54], [48, 66], [40, 80],
        [56, 82], [24, 90], [46, 94], [8, 96], [-4, 86],
    ], float)
    M = len(landmarks)

    # 로봇 제어 시퀀스(완만한 S자; 헤딩을 ±80° 내로 유지 → ±π wrap 경계 회피)
    N = 360
    controls = []
    for k in range(N):
        v = 6.0
        w = 0.0 if k < 40 else (0.2 if k < 110 else (-0.2 if k < 250 else 0.2))
        controls.append((v, w))

    # ---- EKF-SLAM 상태 ----
    dim = 3 + 2 * M
    x = np.zeros(dim)                 # [rx,ry,rθ, l1x,l1y, ...]
    P = np.zeros((dim, dim))
    P[:3, :3] = np.diag([0.01, 0.01, 0.01])   # 로봇 초기 pose 확신
    P[3:, 3:] = np.eye(2 * M) * 1e4           # 랜드마크: 미지(큰 불확실도)
    seen = [False] * M
    seen_count = [0] * M

    # 필터 프로세스 잡음은 실제보다 크게(인플레이션) — EKF-SLAM 과신/일관성 붕괴 방지
    Rmot = np.diag([(3 * V_STD)**2, (3 * W_STD)**2])
    Robs = np.diag([R_STD**2, B_STD**2])
    C_STD = np.radians(2.0)   # 컴퍼스(자력계) 헤딩 측정 잡음 — 현실 로봇의 표준 보조센서

    true_pose = np.array([0.0, 0.0, 0.0])
    odo_pose = np.array([0.0, 0.0, 0.0])   # 순수 오도메트리(관측 미사용) 비교용
    traj_true, traj_est, traj_odo = [], [], []

    for (v, w) in controls:
        # --- 참값 이동 + 오도메트리 잡음 주입 ---
        true_pose = motion(true_pose, v, w)
        v_n = v + rng.normal(0, V_STD)
        w_n = w + rng.normal(0, W_STD)
        odo_pose = motion(odo_pose, v_n, w_n)  # 같은 잡음 제어로 순수 적분

        # --- 예측(로봇 pose만) ---
        th = x[2]
        x[:3] = motion(x[:3], v_n, w_n)
        Gr = np.array([[1, 0, -v_n * DT * np.sin(th)],
                       [0, 1,  v_n * DT * np.cos(th)],
                       [0, 0, 1]])
        # 제어→pose 잡음 매핑
        Vr = np.array([[DT * np.cos(th), 0], [DT * np.sin(th), 0], [0, DT]])
        G = np.eye(dim); G[:3, :3] = Gr
        P = G @ P @ G.T
        P[:3, :3] += Vr @ Rmot @ Vr.T

        # --- 컴퍼스 갱신: 헤딩 직접 관측(관측성 확보) ---
        zc = wrap(true_pose[2] + rng.normal(0, C_STD))
        Hc = np.zeros((1, dim)); Hc[0, 2] = 1.0
        yc = np.array([wrap(zc - x[2])])
        Sc = Hc @ P @ Hc.T + np.array([[C_STD**2]])
        if yc[0]**2 / Sc[0, 0] < 16.0:   # 컴퍼스 이노베이션 게이트
            Kc = P @ Hc.T @ np.linalg.inv(Sc)
            x = x + (Kc @ yc); x[2] = wrap(x[2])
            IKHc = np.eye(dim) - Kc @ Hc
            P = IKHc @ P @ IKHc.T + Kc @ np.array([[C_STD**2]]) @ Kc.T
            P = 0.5 * (P + P.T)

        # --- 관측 & 갱신 ---
        for j in range(M):
            d = landmarks[j] - true_pose[:2]
            r = np.hypot(*d)
            if r > SENSOR_RANGE or r < 1.5:   # 근접은 야코비안 특이 → 스킵
                continue
            seen_count[j] += 1
            z = np.array([r + rng.normal(0, R_STD),
                          wrap(np.arctan2(d[1], d[0]) - true_pose[2] + rng.normal(0, B_STD))])
            li = 3 + 2 * j
            if not seen[j]:
                # 최초 관측: 역관측으로 랜드마크 위치·공분산·상관을 제대로 초기화
                rx, ry, rth = x[:3]
                r, b = z
                cb, sb = np.cos(b + rth), np.sin(b + rth)
                x[li] = rx + r * cb
                x[li + 1] = ry + r * sb
                Grp = np.array([[1, 0, -r * sb], [0, 1, r * cb]])   # ∂L/∂robot
                Gz = np.array([[cb, -r * sb], [sb, r * cb]])         # ∂L/∂z
                P[li:li + 2, li:li + 2] = Grp @ P[:3, :3] @ Grp.T + Gz @ Robs @ Gz.T
                P[li:li + 2, :li] = Grp @ P[:3, :li]                 # 나머지 상태와 상관
                P[:li, li:li + 2] = P[li:li + 2, :li].T
                seen[j] = True
                continue  # 방금 이 관측으로 만든 랜드마크 → 업데이트 생략

            # 예측 관측
            dx, dy = x[li] - x[0], x[li + 1] - x[1]
            q = dx * dx + dy * dy
            sq = np.sqrt(q)
            if sq < 2.0:   # 추정 거리 근접 → 야코비안 특이 → 스킵
                continue
            zhat = np.array([sq, wrap(np.arctan2(dy, dx) - x[2])])
            # 저차원 야코비안 (2 x 5): [rx,ry,rθ, lx,ly]
            Hl = np.array([[-sq * dx, -sq * dy, 0, sq * dx, sq * dy],
                           [dy, -dx, -q, -dy, dx]]) / q
            # 전체 상태로 매핑
            H = np.zeros((2, dim))
            H[:, :3] = Hl[:, :3]
            H[:, li:li + 2] = Hl[:, 3:5]

            y = z - zhat
            y[1] = wrap(y[1])
            S = H @ P @ H.T + Robs
            Sinv = np.linalg.inv(S)
            if y @ Sinv @ y > 16.0:   # 이노베이션 게이트(χ² 2DOF): 이상치 업데이트 거부
                continue
            K = P @ H.T @ Sinv
            x = x + K @ y
            x[2] = wrap(x[2])
            IKH = np.eye(dim) - K @ H
            P = IKH @ P @ IKH.T + K @ Robs @ K.T   # Joseph 형태(수치 안정)
            P = 0.5 * (P + P.T)                     # 대칭 유지

        traj_true.append(true_pose[:2].copy())
        traj_est.append(x[:2].copy())
        traj_odo.append(odo_pose[:2].copy())

    traj_true = np.array(traj_true); traj_est = np.array(traj_est); odo_traj = np.array(traj_odo)
    lm_est = x[3:].reshape(-1, 2)

    traj_rmse = float(np.sqrt(np.mean(np.sum((traj_est - traj_true) ** 2, axis=1))))
    odo_rmse = float(np.sqrt(np.mean(np.sum((odo_traj - traj_true) ** 2, axis=1))))
    lm_err_all = np.sqrt(np.sum((lm_est - landmarks) ** 2, axis=1))
    seen_mask = np.array(seen)
    lm_err = lm_err_all[seen_mask]  # 관측된 랜드마크만
    print("=== EKF-SLAM 결과 ===")
    print(f"로봇 궤적 RMSE      : {traj_rmse:.3f} m")
    print(f"관측된 랜드마크 {seen_mask.sum()}/{M}개 위치오차: mean={lm_err.mean():.3f} m  max={lm_err.max():.3f} m")
    print(f"순수 오도메트리 RMSE: {odo_rmse:.3f} m  → SLAM이 {odo_rmse/traj_rmse:.1f}배 개선")

    if not plot:
        return traj_rmse, odo_rmse, float(lm_err.mean())

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(traj_true[:, 0], traj_true[:, 1], "g-", lw=2, label="true trajectory")
    ax.plot(odo_traj[:, 0], odo_traj[:, 1], "r-", lw=1, alpha=0.6, label="odometry only (drifts)")
    ax.plot(traj_est[:, 0], traj_est[:, 1], "b-", lw=1.4, label="EKF-SLAM estimate")
    ax.plot(landmarks[seen_mask, 0], landmarks[seen_mask, 1], "g*", ms=15, label="true landmarks")
    ax.plot(lm_est[seen_mask, 0], lm_est[seen_mask, 1], "bx", ms=9, label="estimated landmarks")
    ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"EKF-SLAM: traj RMSE {traj_rmse:.2f}m, map err {lm_err.mean():.2f}m "
                 f"(odometry-only {odo_rmse:.1f}m)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(Path("outputs") / "05_ekf_slam.png", dpi=130)
    print("\n[plot] outputs/05_ekf_slam.png")
    return traj_rmse, odo_rmse, float(lm_err.mean())


if __name__ == "__main__":
    main()
