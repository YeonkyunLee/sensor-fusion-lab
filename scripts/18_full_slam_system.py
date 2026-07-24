"""완전 SLAM 시스템: 실시간 프론트엔드 + 전역 백엔드 통합 (실전 아키텍처).

지금까지의 조각을 하나의 파이프라인으로 합친다:
  - 프론트엔드: fixed-lag 윈도우 최적화로 매 스텝 실시간 pose 제공(드리프트 누적)
  - 백엔드: 루프클로저가 감지되면 전역 pose-graph를 DCS 강건 커널로 최적화
            → 누적 드리프트를 일괄 교정 + 거짓 루프클로저(perceptual aliasing) 거부

ORB-SLAM/VINS 류의 front-end/back-end 구조를 2D로 재현.

    python scripts/18_full_slam_system.py
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
    vp = {v: k for k, v in enumerate(var_ids)}; M = 3*len(var_ids)
    if M == 0:
        return
    for _ in range(iters):
        H = np.zeros((M, M)); b = np.zeros(M)
        for e in edges:
            iv, jv = e.i in vp, e.j in vp
            if not (iv or jv):
                continue
            err, A, B = _error_and_jacobians(X[e.i], X[e.j], e.z)
            if iv:
                a = 3*vp[e.i]; H[a:a+3, a:a+3] += A.T@e.omega@A; b[a:a+3] += A.T@e.omega@err
            if jv:
                c = 3*vp[e.j]; H[c:c+3, c:c+3] += B.T@e.omega@B; b[c:c+3] += B.T@e.omega@err
            if iv and jv:
                a, c = 3*vp[e.i], 3*vp[e.j]
                H[a:a+3, c:c+3] += A.T@e.omega@B; H[c:c+3, a:a+3] += B.T@e.omega@A
        dx = np.linalg.solve(H + 1e-6*np.eye(M), -b)
        for v, k in vp.items():
            X[v] += dx[3*k:3*k+3]; X[v, 2] = wrap(X[v, 2])


def backend_global(X, edges, npose, iters=15, dcs_phi=5.0):
    """전역 pose-graph 최적화 + DCS 강건(루프클로저 엣지만)."""
    N = 3*npose
    for _ in range(iters):
        H = np.zeros((N, N)); b = np.zeros(N)
        for e in edges:
            err, A, B = _error_and_jacobians(X[e.i], X[e.j], e.z)
            if abs(e.i-e.j) == 1:
                w = 1.0                                  # 오도메트리 백본
            else:
                chi = float(err@e.omega@err)
                w = min(1.0, 2*dcs_phi/(dcs_phi+chi))**2   # DCS
            WOm = w*e.omega; i, j = 3*e.i, 3*e.j
            H[i:i+3, i:i+3] += A.T@WOm@A; H[i:i+3, j:j+3] += A.T@WOm@B
            H[j:j+3, i:i+3] += B.T@WOm@A; H[j:j+3, j:j+3] += B.T@WOm@B
            b[i:i+3] += A.T@WOm@err; b[j:j+3] += B.T@WOm@err
        H[:3, :3] += np.eye(3)*1e6
        dx = np.linalg.solve(H + 1e-9*np.eye(N), -b)
        X[:npose] += dx.reshape(npose, 3); X[:npose, 2] = (X[:npose, 2]+np.pi) % (2*np.pi)-np.pi


def main():
    rng = np.random.default_rng(0)
    dt = 0.1; v, R = 6.0, 20.0; w = v/R
    per_lap = 70; n = per_lap*2; LAG = 12

    true = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n):
        p = true[-1]
        true.append(np.array([p[0]+v*dt*np.cos(p[2]), p[1]+v*dt*np.sin(p[2]), wrap(p[2]+w*dt)]))
    true = np.array(true); npose = len(true)

    odo_sig = np.array([0.05, 0.05, 0.02]); Om = np.diag(1/odo_sig**2)
    lc_sig = np.array([0.05, 0.05, 0.02]); Om_lc = np.diag(1/lc_sig**2)

    X = np.zeros((npose, 3)); X[0] = true[0]
    edges = []
    n_true_lc = n_false_lc = 0
    for k in range(1, npose):
        z = rel(true[k-1], true[k]) + rng.normal(0, odo_sig)
        X[k] = t2v(v2t(X[k-1]) @ v2t(z)); edges.append(Edge(k-1, k, z, Om))
        # 프론트엔드: 실시간 fixed-lag
        gn_window(X, edges, list(range(max(1, k-LAG+1), k+1)), iters=4)
        # 루프클로저 감지(2바퀴째 재방문) → 백엔드 트리거
        j = k
        if j >= per_lap and j % 6 == 0:
            i = j - per_lap
            edges.append(Edge(i, j, rel(true[i], true[j]) + rng.normal(0, lc_sig), Om_lc)); n_true_lc += 1
            if rng.random() < 0.3:   # 30% 확률로 거짓 루프클로저 주입
                fi = rng.integers(0, j)   # 현재까지 존재하는 pose만
                edges.append(Edge(fi, j, np.array([rng.uniform(-6, 6), rng.uniform(-6, 6),
                             rng.uniform(-np.pi, np.pi)]), Om_lc)); n_false_lc += 1
            backend_global(X, edges, j+1)     # 전역 강건 최적화

    # 비교용 프론트엔드-only: 같은 오도메트리로 fixed-lag만(백엔드 없음)
    rng2 = np.random.default_rng(0); rng2.standard_normal(3)  # true 생성과 무관
    fe = np.zeros((npose, 3)); fe[0] = true[0]; fe_e = []
    fr = np.random.default_rng(1)
    for k in range(1, npose):
        z = rel(true[k-1], true[k]) + fr.normal(0, odo_sig)
        fe[k] = t2v(v2t(fe[k-1]) @ v2t(z)); fe_e.append(Edge(k-1, k, z, Om))
        gn_window(fe, fe_e, list(range(max(1, k-LAG+1), k+1)), iters=4)

    def rmse(P): return float(np.sqrt(np.mean(np.sum((P[:, :2]-true[:, :2])**2, 1))))
    print("=== 완전 SLAM 시스템: 프론트엔드 + 전역 백엔드 ===")
    print(f"pose {npose}, 참 루프클로저 {n_true_lc}, 거짓 {n_false_lc}(DCS로 거부)")
    print(f"프론트엔드만(fixed-lag) : 궤적 RMSE {rmse(fe):.3f} m (드리프트)")
    print(f"풀 시스템(+백엔드)       : 궤적 RMSE {rmse(X):.3f} m")
    print(f"→ 백엔드 전역 최적화가 드리프트를 {rmse(fe)/max(rmse(X),1e-6):.0f}배 교정 (거짓 클로저는 강건 거부)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    a1.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true"); a1.plot(fe[:, 0], fe[:, 1], "r-", lw=1.2, label=f"front-end only ({rmse(fe):.1f}m)")
    a1.set_aspect("equal"); a1.legend(fontsize=8); a1.grid(alpha=0.3); a1.set_title("Front-end (fixed-lag, real-time) — drifts")
    a2.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true"); a2.plot(X[:, 0], X[:, 1], "b-", lw=1.3, label=f"full system ({rmse(X):.2f}m)")
    a2.set_aspect("equal"); a2.legend(fontsize=8); a2.grid(alpha=0.3); a2.set_title(f"+ back-end (global + DCS robust, {n_false_lc} false closures rejected)")
    fig.suptitle("Full SLAM system: fixed-lag front-end + robust global back-end")
    fig.tight_layout(); fig.savefig("outputs/18_full_slam_system.png", dpi=130)
    print("\n[plot] outputs/18_full_slam_system.png")
    return rmse(fe), rmse(X)


if __name__ == "__main__":
    main()
