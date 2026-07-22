"""그래프 SLAM: 루프 클로저로 전체 궤적을 한 번에 정렬.

로봇이 루프를 돈다. 오도메트리(잡음 상대이동)만 적분하면 궤적이 벌어진다(open loop).
pose-graph에 오도메트리 엣지 + 루프클로저 엣지(마지막≈처음)를 넣고 최적화하면,
EKF와 달리 **과거 전체가 재선형화**되어 루프가 닫히고 참 궤적에 정렬된다.

    python scripts/07_pose_graph_slam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import Edge, optimize, t2v, v2t, wrap  # noqa: E402


def rel(a, b):
    """pose a 기준 pose b의 상대 pose (a→b)."""
    return t2v(np.linalg.inv(v2t(a)) @ v2t(b))


def main():
    rng = np.random.default_rng(0)
    dt = 0.1
    v, R = 6.0, 22.0
    w = v / R
    n = int(2 * np.pi / (w * dt))  # 정확히 한 바퀴

    # 참 pose 시퀀스(원 루프, 시작=끝 근처)
    true = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n):
        p = true[-1]
        true.append(np.array([p[0] + v*dt*np.cos(p[2]), p[1] + v*dt*np.sin(p[2]), wrap(p[2] + w*dt)]))
    true = np.array(true)
    npose = len(true)

    # 오도메트리 엣지(참 상대이동 + 잡음) → 적분하면 드리프트
    odom_sigma = np.array([0.04, 0.04, 0.012])
    info_odo = np.diag(1.0 / odom_sigma**2)
    edges = []
    x_odo = [true[0].copy()]
    for k in range(n):
        z = rel(true[k], true[k+1]) + rng.normal(0, odom_sigma)
        edges.append(Edge(k, k+1, z, info_odo))
        # 오도메트리만으로 초기 추정치 누적
        x_odo.append(t2v(v2t(x_odo[-1]) @ v2t(z)))
    x_odo = np.array(x_odo)

    # 루프 클로저 엣지: 마지막 pose가 처음 pose 근처임을 인식(정밀 측정)
    lc_sigma = np.array([0.05, 0.05, 0.02])
    z_lc = rel(true[0], true[-1]) + rng.normal(0, lc_sigma)
    edges.append(Edge(0, npose - 1, z_lc, np.diag(1.0 / lc_sigma**2)))

    # 최적화
    x_opt, hist = optimize(x_odo, edges, iters=30)

    def rmse(X):
        return float(np.sqrt(np.mean(np.sum((X[:, :2] - true[:, :2])**2, axis=1))))

    print("=== 그래프 SLAM (pose-graph) ===")
    print(f"pose {npose}개, 엣지 {len(edges)}개(오도메트리 {n} + 루프클로저 1)")
    print(f"오도메트리만    : 궤적 RMSE={rmse(x_odo):.3f} m,  종료 gap={np.hypot(*(x_odo[-1,:2]-x_odo[0,:2])):.3f} m")
    print(f"그래프 최적화 후: 궤적 RMSE={rmse(x_opt):.3f} m,  종료 gap={np.hypot(*(x_opt[-1,:2]-x_opt[0,:2])):.3f} m")
    print(f"→ 궤적 오차를 {rmse(x_odo)/max(rmse(x_opt),1e-6):.0f}배 줄임")
    print(f"chi2: {hist[0]:.1f} → {hist[-1]:.3g} ({len(hist)} iters)")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true loop")
    ax.plot(x_odo[:, 0], x_odo[:, 1], "r--", lw=1.3, label=f"odometry only (open, {rmse(x_odo):.1f}m)")
    ax.plot(x_opt[:, 0], x_opt[:, 1], "b-", lw=1.5, label=f"pose-graph optimized ({rmse(x_opt):.2f}m)")
    ax.plot(true[0, 0], true[0, 1], "ko", ms=9, label="start/end")
    ax.set_aspect("equal"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Graph SLAM: one loop-closure edge snaps the whole trajectory shut")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(Path("outputs") / "07_pose_graph_slam.png", dpi=130)
    print("\n[plot] outputs/07_pose_graph_slam.png")
    return rmse(x_odo), rmse(x_opt)


if __name__ == "__main__":
    main()
