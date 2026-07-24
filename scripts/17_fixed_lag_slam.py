"""온라인 SLAM: fixed-lag smoother vs 전체 배치 최적화.

실전 온라인 추정(VIO 등)은 매 스텝 전체 궤적을 다시 풀 수 없다. **fixed-lag smoother**는
최근 L개 pose만 변수로 두고 나머지는 고정 → 스텝당 문제 크기가 일정(O(1))하다. 대신
윈도우 밖 과거는 재조정 못 해 전역 일관성은 전체 배치보다 약하다 — 속도 vs 일관성의
정직한 트레이드오프.

    python scripts/17_fixed_lag_slam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import Edge, _error_and_jacobians, t2v, v2t, wrap  # noqa: E402


def rel(a, b):
    return t2v(np.linalg.inv(v2t(a)) @ v2t(b))


def gn_window(X, edges, var_ids, iters=5):
    """var_ids 의 pose만 최적화(나머지는 고정 상수). 반환: 갱신된 X."""
    vpos = {v: k for k, v in enumerate(var_ids)}
    M = 3 * len(var_ids)
    if M == 0:
        return X
    for _ in range(iters):
        H = np.zeros((M, M)); b = np.zeros(M)
        for e in edges:
            iv, jv = e.i in vpos, e.j in vpos
            if not (iv or jv):
                continue
            err, A, B = _error_and_jacobians(X[e.i], X[e.j], e.z)
            if iv:
                a = 3*vpos[e.i]; H[a:a+3, a:a+3] += A.T@e.omega@A; b[a:a+3] += A.T@e.omega@err
            if jv:
                c = 3*vpos[e.j]; H[c:c+3, c:c+3] += B.T@e.omega@B; b[c:c+3] += B.T@e.omega@err
            if iv and jv:
                a, c = 3*vpos[e.i], 3*vpos[e.j]
                H[a:a+3, c:c+3] += A.T@e.omega@B; H[c:c+3, a:a+3] += B.T@e.omega@A
        H += 1e-6*np.eye(M)
        dx = np.linalg.solve(H, -b)
        for v, k in vpos.items():
            X[v] += dx[3*k:3*k+3]; X[v, 2] = wrap(X[v, 2])
    return X


def main():
    rng = np.random.default_rng(0)
    dt = 0.1; v, R = 6.0, 20.0; w = v/R
    per_lap = 70; n = per_lap*2
    LAG = 15

    true = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n):
        p = true[-1]
        true.append(np.array([p[0]+v*dt*np.cos(p[2]), p[1]+v*dt*np.sin(p[2]), wrap(p[2]+w*dt)]))
    true = np.array(true); npose = len(true)

    odo_sig = np.array([0.05, 0.05, 0.02]); Om = np.diag(1/odo_sig**2)
    lc_sig = np.array([0.05, 0.05, 0.02]); Om_lc = np.diag(1/lc_sig**2)
    odo = {k: (k-1, k, rel(true[k-1], true[k]) + rng.normal(0, odo_sig)) for k in range(1, npose)}
    loops = {}
    for j in range(per_lap, npose):
        i = j-per_lap
        if j % 4 == 0:
            loops.setdefault(j, []).append((i, j, rel(true[i], true[j]) + rng.normal(0, lc_sig)))

    def run(fixed_lag):
        X = np.zeros((npose, 3)); X[0] = true[0]; edges = []
        sizes = []
        for k in range(1, npose):
            i, j, z = odo[k]
            X[k] = t2v(v2t(X[k-1]) @ v2t(z)); edges.append(Edge(i, j, z, Om))
            for (li, lj, lz) in loops.get(k, []):
                edges.append(Edge(li, lj, lz, Om_lc))
            if fixed_lag:
                var_ids = list(range(max(1, k-LAG+1), k+1))     # 최근 LAG개만 변수
            else:
                var_ids = list(range(1, k+1))                    # 전체
            gn_window(X, edges, var_ids, iters=5)
            sizes.append(len(var_ids))
        return X, np.array(sizes)

    Xf, sf = run(fixed_lag=True)
    Xb, sb = run(fixed_lag=False)

    def rmse(X): return float(np.sqrt(np.mean(np.sum((X[:, :2]-true[:, :2])**2, 1))))
    print("=== 온라인 SLAM: fixed-lag smoother vs 전체 배치 ===")
    print(f"pose {npose}, 루프클로저 {sum(len(v) for v in loops.values())}, lag={LAG}")
    print(f"fixed-lag : 스텝당 문제크기 일정(최대 {sf.max()} pose),  최종 RMSE {rmse(Xf):.3f} m")
    print(f"전체 배치 : 스텝당 문제크기 증가(최대 {sb.max()} pose),  최종 RMSE {rmse(Xb):.3f} m")
    print(f"→ fixed-lag는 스텝당 O(1) 비용(정확도는 전역 일관성이 약해 {rmse(Xf)/max(rmse(Xb),1e-6):.1f}x)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(range(1, npose), sb*3, "r-", label="full batch (grows O(N))")
    ax1.plot(range(1, npose), sf*3, "b-", label=f"fixed-lag L={LAG} (constant)")
    ax1.set_xlabel("poses processed"); ax1.set_ylabel("solve dimension (per step)")
    ax1.set_title("Per-step cost: fixed-lag O(1) vs batch O(N)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true")
    ax2.plot(Xf[:, 0], Xf[:, 1], "b-", lw=1.2, label=f"fixed-lag ({rmse(Xf):.2f}m)")
    ax2.plot(Xb[:, 0], Xb[:, 1], "r--", lw=1.0, alpha=0.7, label=f"full batch ({rmse(Xb):.2f}m)")
    ax2.set_aspect("equal"); ax2.set_title("Online estimates"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle("Fixed-lag smoother vs full-batch online SLAM (speed vs global consistency)")
    fig.tight_layout(); fig.savefig("outputs/17_fixed_lag_slam.png", dpi=130)
    print("\n[plot] outputs/17_fixed_lag_slam.png")
    return rmse(Xf), rmse(Xb), int(sf.max()), int(sb.max())


if __name__ == "__main__":
    main()
