"""거짓 루프 클로저에 강건한 그래프 SLAM (Huber 커널).

실전 SLAM의 골칫거리: 장소 인식이 비슷한 곳을 혼동해 **틀린 루프 클로저**를 넣는다
(perceptual aliasing). 단 몇 개의 거짓 제약이 최소제곱 최적화를 통째로 오염시킨다.
Huber 강건 커널이 이상치 엣지를 반복 재가중으로 걷어내는 것을 보인다.

    python scripts/11_robust_slam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import Edge, optimize, optimize_robust, t2v, v2t, wrap  # noqa: E402


def rel(a, b):
    return t2v(np.linalg.inv(v2t(a)) @ v2t(b))


def main():
    rng = np.random.default_rng(2)
    dt = 0.1
    v, R = 6.0, 20.0
    w = v / R
    n = int(2 * np.pi / (w * dt))

    true = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n):
        p = true[-1]
        true.append(np.array([p[0]+v*dt*np.cos(p[2]), p[1]+v*dt*np.sin(p[2]), wrap(p[2]+w*dt)]))
    true = np.array(true); npose = len(true)

    odo_sigma = np.array([0.04, 0.04, 0.012]); info = np.diag(1.0/odo_sigma**2)
    edges = []; x0 = [true[0].copy()]
    for k in range(n):
        z = rel(true[k], true[k+1]) + rng.normal(0, odo_sigma)
        edges.append(Edge(k, k+1, z, info)); x0.append(t2v(v2t(x0[-1]) @ v2t(z)))
    x0 = np.array(x0)

    lc_sigma = np.array([0.05, 0.05, 0.02]); lc_info = np.diag(1.0/lc_sigma**2)
    # 참 루프 클로저(마지막↔처음)
    edges.append(Edge(0, npose-1, rel(true[0], true[-1]) + rng.normal(0, lc_sigma), lc_info))
    n_true_lc = 1
    # 거짓 루프 클로저 3개: 서로 무관한 pose 쌍을 잘못 연결
    false_pairs = [(30, 150), (60, 200), (100, 20)]
    for (i, j) in false_pairs:
        z_false = rel(true[i], true[j]) + np.array([8.0, -6.0, 1.0])  # 크게 틀린 측정
        edges.append(Edge(i, j, z_false, lc_info))

    x_naive, _ = optimize(x0.copy(), edges, iters=30)
    x_robust, weights = optimize_robust(x0.copy(), edges, iters=40, huber_delta=2.5)
    # 하드 리젝션: 오도메트리(체인)는 항상 유지, 루프클로저 중 저가중치만 제거
    inliers = [e for e, w_ in zip(edges, weights)
               if abs(e.j - e.i) == 1 or w_ > 0.5]
    x_clean, _ = optimize(x0.copy(), inliers, iters=30)
    n_rejected = len(edges) - len(inliers)

    def rmse(X):
        return float(np.sqrt(np.mean(np.sum((X[:, :2]-true[:, :2])**2, axis=1))))

    w_false = [weights[n + 1 + k] for k in range(len(false_pairs))]
    w_true = weights[n]

    print("=== 거짓 루프 클로저에 대한 강건성 ===")
    print(f"참 루프클로저 {n_true_lc}개 + 거짓 {len(false_pairs)}개 주입")
    print(f"naive(최소제곱)      : 궤적 RMSE={rmse(x_naive):.3f} m  (거짓 제약에 오염)")
    print(f"robust(Huber)        : 궤적 RMSE={rmse(x_robust):.3f} m")
    print(f"robust+리젝션 재최적화: 궤적 RMSE={rmse(x_clean):.3f} m  (이상치 {n_rejected}개 제거)")
    print(f"→ naive 대비 {rmse(x_naive)/max(rmse(x_clean),1e-6):.0f}배 개선")
    print(f"거짓 엣지 가중치: {[round(x,3) for x in w_false]} (↓걸러짐), 참 엣지: {w_true:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, X, title in [(ax1, x_naive, f"naive least-squares ({rmse(x_naive):.1f}m)"),
                         (ax2, x_clean, f"robust + rejection ({rmse(x_clean):.2f}m)")]:
        ax.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true")
        ax.plot(X[:, 0], X[:, 1], "b-", lw=1.3, label="optimized")
        for (i, j) in false_pairs:
            ax.plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], "r-", lw=0.8, alpha=0.6)
        ax.plot([], [], "r-", label="false loop closures")
        ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title(title)
    fig.suptitle("Robust back-end rejects false loop closures (perceptual aliasing)")
    fig.tight_layout(); fig.savefig(Path("outputs") / "11_robust_slam.png", dpi=130)
    print("\n[plot] outputs/11_robust_slam.png")
    return rmse(x_naive), rmse(x_clean)


if __name__ == "__main__":
    main()
