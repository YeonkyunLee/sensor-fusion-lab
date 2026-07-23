"""현대 SLAM 아키텍처: VIO 프론트엔드 + 팩터그래프 백엔드.

실제 SLAM 시스템의 구조를 그대로 재현한다:
  - 프론트엔드(VIO): 키프레임 간 상대 pose(오도메트리)를 추정 — 드리프트 누적
  - 백엔드(pose-graph): 오도메트리 팩터 + 루프클로저 팩터를 함께 최적화

로봇이 원을 두 바퀴 돈다. VIO는 2바퀴에 걸쳐 드리프트한다. 2바퀴째가 1바퀴째의
같은 장소를 재방문(place recognition)하면 루프클로저 팩터가 생겨, 백엔드가 전체
궤적을 하나의 깨끗한 원으로 정렬한다.

    python scripts/10_vio_graph_slam.py
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
    return t2v(np.linalg.inv(v2t(a)) @ v2t(b))


def main():
    rng = np.random.default_rng(1)
    dt = 0.1
    v, R = 6.0, 20.0
    w = v / R
    per_lap = int(2 * np.pi / (w * dt))
    laps = 2
    n = per_lap * laps

    # 참 pose (원 2바퀴; 2바퀴째는 1바퀴째와 겹침)
    true = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n):
        p = true[-1]
        true.append(np.array([p[0] + v*dt*np.cos(p[2]), p[1] + v*dt*np.sin(p[2]), wrap(p[2] + w*dt)]))
    true = np.array(true)
    npose = len(true)

    # 프론트엔드(VIO): 연속 키프레임 상대 pose + VIO 잡음
    vio_sigma = np.array([0.05, 0.05, 0.015])
    info_odo = np.diag(1.0 / vio_sigma**2)
    edges = []
    x_vio = [true[0].copy()]
    for k in range(n):
        z = rel(true[k], true[k+1]) + rng.normal(0, vio_sigma)
        edges.append(Edge(k, k+1, z, info_odo))
        x_vio.append(t2v(v2t(x_vio[-1]) @ v2t(z)))
    x_vio = np.array(x_vio)

    # 루프클로저: 2바퀴째 키프레임이 1바퀴째의 같은 장소를 재방문 → 팩터 추가
    lc_sigma = np.array([0.08, 0.08, 0.03])
    info_lc = np.diag(1.0 / lc_sigma**2)
    n_lc = 0
    for j in range(per_lap, npose):        # 2바퀴째
        i = j - per_lap                     # 대응하는 1바퀴째
        if i < 0:
            continue
        if np.hypot(*(true[j, :2] - true[i, :2])) < 1.0 and j % 5 == 0:  # 근접 + 간헐 검출
            z = rel(true[i], true[j]) + rng.normal(0, lc_sigma)
            edges.append(Edge(i, j, z, info_lc))
            n_lc += 1

    x_opt, hist = optimize(x_vio, edges, iters=40)

    def rmse(X):
        return float(np.sqrt(np.mean(np.sum((X[:, :2] - true[:, :2])**2, axis=1))))

    print("=== VIO 프론트엔드 + 팩터그래프 백엔드 (현대 SLAM) ===")
    print(f"키프레임 {npose}, 오도메트리 팩터 {n}, 루프클로저 팩터 {n_lc}")
    print(f"VIO 프론트엔드만 : 궤적 RMSE={rmse(x_vio):.3f} m (2바퀴 드리프트)")
    print(f"+ 그래프 백엔드   : 궤적 RMSE={rmse(x_opt):.3f} m")
    print(f"→ 백엔드가 드리프트를 {rmse(x_vio)/max(rmse(x_opt),1e-6):.0f}배 줄임")
    print(f"chi2: {hist[0]:.0f} → {hist[-1]:.3g} ({len(hist)} iters)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    ax1.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true (2 laps)")
    ax1.plot(x_vio[:, 0], x_vio[:, 1], "r-", lw=1.2, label=f"VIO front-end ({rmse(x_vio):.1f}m)")
    ax1.set_aspect("equal"); ax1.legend(); ax1.grid(alpha=0.3); ax1.set_title("Front-end only: VIO drifts over 2 laps")
    ax2.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true")
    ax2.plot(x_opt[:, 0], x_opt[:, 1], "b-", lw=1.4, label=f"VIO + graph ({rmse(x_opt):.2f}m)")
    # 루프클로저 팩터 시각화
    for e in edges:
        if abs(e.j - e.i) > 10:
            ax2.plot([x_opt[e.i, 0], x_opt[e.j, 0]], [x_opt[e.i, 1], x_opt[e.j, 1]],
                     "c-", lw=0.3, alpha=0.5)
    ax2.set_aspect("equal"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title(f"+ graph back-end: {n_lc} loop closures align it")
    fig.suptitle("Modern SLAM: VIO front-end + factor-graph back-end")
    fig.tight_layout(); fig.savefig(Path("outputs") / "10_vio_graph_slam.png", dpi=130)
    print("\n[plot] outputs/10_vio_graph_slam.png")
    return rmse(x_vio), rmse(x_opt)


if __name__ == "__main__":
    main()
