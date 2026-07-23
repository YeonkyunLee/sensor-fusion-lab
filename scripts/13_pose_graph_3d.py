"""3D SE(3) pose-graph SLAM: 기울어진 원 2바퀴 + 루프 클로저.

실전 로봇/드론은 3D다. 3D 오도메트리는 위치·자세 모두 드리프트한다. 나선을 오르내리며
시작점 근처로 돌아와 루프 클로저 팩터를 추가하면, SE(3) pose-graph 최적화가 3D 궤적
전체를 정렬한다.

    python scripts/13_pose_graph_3d.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph3d import Edge3D, optimize_se3  # noqa: E402
from sensor_fusion.se3 import se3_exp, se3_inv, se3_log  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    per_lap = 90
    laps = 2
    n = per_lap * laps

    # 참 3D 궤적: 기울어진 평면 위의 원을 두 바퀴(같은 3D 경로 → 랩 간 재방문)
    tilt = 0.5  # 평면 기울기 [rad]
    ct, st = np.cos(tilt), np.sin(tilt)
    Rtilt = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    rad = 14.0
    poses_true = []
    for k in range(n + 1):
        t = 2 * np.pi * (k % per_lap) / per_lap
        p = Rtilt @ np.array([rad * np.cos(t), rad * np.sin(t), 0.0])
        yaw = t + np.pi / 2
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        T = np.eye(4); T[:3, :3] = Rtilt @ Rz; T[:3, 3] = p
        poses_true.append(T)

    # 오도메트리: 참 상대 pose + se(3) 잡음 → 적분하면 드리프트
    odo_sig = np.array([0.03, 0.03, 0.03, 0.01, 0.01, 0.02])
    Om = np.diag(1.0 / odo_sig**2)
    edges = []
    X0 = [poses_true[0].copy()]
    for k in range(n):
        Z = se3_inv(poses_true[k]) @ poses_true[k+1]
        Zn = Z @ se3_exp(rng.normal(0, odo_sig))
        edges.append(Edge3D(k, k+1, Zn, Om))
        X0.append(X0[-1] @ Zn)

    # 루프 클로저: 2바퀴째가 1바퀴째의 같은 3D 지점 재방문 → 다중 팩터
    lc_sig = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.03])
    Om_lc = np.diag(1.0 / lc_sig**2); n_lc = 0
    for j in range(per_lap, n + 1):
        i = j - per_lap
        if j % 4 == 0:  # 간헐 검출
            Z = (se3_inv(poses_true[i]) @ poses_true[j]) @ se3_exp(rng.normal(0, lc_sig))
            edges.append(Edge3D(i, j, Z, Om_lc)); n_lc += 1

    X_opt, hist = optimize_se3(X0, edges, iters=25)

    def pos(Xs): return np.array([T[:3, 3] for T in Xs])
    tp = pos(poses_true); op = pos(X0); xp = pos(X_opt)

    def rmse(P): return float(np.sqrt(np.mean(np.sum((P - tp)**2, axis=1))))
    print("=== 3D SE(3) pose-graph SLAM ===")
    print(f"pose {len(poses_true)}, 오도메트리 팩터 {n}, 루프클로저 {n_lc}")
    print(f"오도메트리만 : 3D 위치 RMSE={rmse(op):.3f} m")
    print(f"SE(3) 최적화 : 3D 위치 RMSE={rmse(xp):.3f} m")
    print(f"→ {rmse(op)/max(rmse(xp),1e-6):.0f}배 개선,  chi2 {hist[0]:.0f} → {hist[-1]:.3g} ({len(hist)} iters)")

    fig = plt.figure(figsize=(13, 6))
    for idx, (P, title) in enumerate([(op, f"odometry ({rmse(op):.1f}m)"),
                                      (xp, f"SE(3) optimized ({rmse(xp):.2f}m)")]):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], "g-", lw=2.5, label="true")
        ax.plot(P[:, 0], P[:, 1], P[:, 2], "b-", lw=1.3, label="estimate")
        ax.scatter(*tp[0], c="k", s=40); ax.set_title(title); ax.legend(fontsize=8)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.suptitle("3D SE(3) pose-graph SLAM: tilted circle (2 laps) + loop closures")
    fig.tight_layout(); fig.savefig(Path("outputs") / "13_pose_graph_3d.png", dpi=130)
    print("\n[plot] outputs/13_pose_graph_3d.png")
    return rmse(op), rmse(xp)


if __name__ == "__main__":
    main()
