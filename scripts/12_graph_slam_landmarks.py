"""완전 그래프 SLAM — pose + 랜드마크 공동 최적화 (2D bundle adjustment).

지금까지의 pose-graph는 pose만 노드로 뒀다. 여기서는 **랜드마크도 노드**로 넣어,
오도메트리 팩터(pose-pose) + range-bearing 관측 팩터(pose-landmark)를 하나의 큰
최소제곱으로 함께 푼다 — EKF-SLAM(순차 필터)의 배치(batch) 버전이자 현대 SLAM
백엔드(BA)의 2D 형태.

    python scripts/12_graph_slam_landmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import _error_and_jacobians, t2v, v2t, wrap  # noqa: E402

DT = 0.1
SENSOR_RANGE = 22.0


def obs_factor(pose, lm, z, Om):
    """range-bearing 관측 팩터: 오차 e(2), J_pose(2x3), J_lm(2x2)."""
    dx, dy = lm[0] - pose[0], lm[1] - pose[1]
    q = dx*dx + dy*dy; r = np.sqrt(q)
    h = np.array([r, wrap(np.arctan2(dy, dx) - pose[2])])
    e = np.array([h[0] - z[0], wrap(h[1] - z[1])])
    Jp = np.array([[-dx/r, -dy/r, 0.0], [dy/q, -dx/q, -1.0]])
    Jl = np.array([[dx/r, dy/r], [-dy/q, dx/q]])
    return e, Jp, Jl, Om


def optimize(poses, lms, odo_edges, obs, iters=25):
    """poses (P,3), lms (L,2), odo_edges[(i,j,z,Om)], obs[(p,l,z,Om)]."""
    P, L = len(poses), len(lms)
    x = np.concatenate([poses.reshape(-1), lms.reshape(-1)])
    N = 3*P + 2*L

    def pslice_p(i): return slice(3*i, 3*i+3)
    def slice_l(l): return slice(3*P + 2*l, 3*P + 2*l + 2)

    hist = []
    for _ in range(iters):
        H = np.zeros((N, N)); b = np.zeros(N); chi = 0.0
        # 오도메트리 팩터
        for (i, j, z, Om) in odo_edges:
            xi, xj = x[pslice_p(i)], x[pslice_p(j)]
            e, A, B = _error_and_jacobians(xi, xj, z)
            chi += e @ Om @ e
            H[pslice_p(i), pslice_p(i)] += A.T @ Om @ A
            H[pslice_p(i), pslice_p(j)] += A.T @ Om @ B
            H[pslice_p(j), pslice_p(i)] += B.T @ Om @ A
            H[pslice_p(j), pslice_p(j)] += B.T @ Om @ B
            b[pslice_p(i)] += A.T @ Om @ e
            b[pslice_p(j)] += B.T @ Om @ e
        # 관측 팩터
        for (p, l, z, Om) in obs:
            e, Jp, Jl, _ = obs_factor(x[pslice_p(p)], x[slice_l(l)], z, Om)
            chi += e @ Om @ e
            sp, sl = pslice_p(p), slice_l(l)
            H[sp, sp] += Jp.T @ Om @ Jp
            H[sp, sl] += Jp.T @ Om @ Jl
            H[sl, sp] += Jl.T @ Om @ Jp
            H[sl, sl] += Jl.T @ Om @ Jl
            b[sp] += Jp.T @ Om @ e
            b[sl] += Jl.T @ Om @ e
        H[0:3, 0:3] += np.eye(3) * 1e6      # pose 0 고정(게이지)
        dx = np.linalg.solve(H + 1e-9*np.eye(N), -b)
        x += dx
        for i in range(P):
            x[3*i+2] = wrap(x[3*i+2])
        hist.append(float(chi))
        if np.max(np.abs(dx)) < 1e-4:
            break
    return x[:3*P].reshape(P, 3), x[3*P:].reshape(L, 2), hist


def main():
    rng = np.random.default_rng(0)
    v, R = 6.0, 20.0; w = v/R
    n = int(2*np.pi/(w*DT))

    # 참 pose (한 바퀴) + 참 랜드마크
    true = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n):
        p = true[-1]
        true.append(np.array([p[0]+v*DT*np.cos(p[2]), p[1]+v*DT*np.sin(p[2]), wrap(p[2]+w*DT)]))
    true = np.array(true); P = len(true)
    ang = np.linspace(0, 2*np.pi, 11)[:-1]
    lms_true = np.stack([R + (R+7)*np.cos(ang-np.pi/2), R + (R+7)*np.sin(ang-np.pi/2)], 1)
    # 실제 원 중심은 (0,R); 랜드마크를 그 주변에
    lms_true = np.stack([(R+7)*np.cos(ang-np.pi/2), R + (R+7)*np.sin(ang-np.pi/2)], 1)
    L = len(lms_true)

    # 오도메트리(잡음) → 초기 pose 추정(드리프트)
    odo_sig = np.array([0.05, 0.05, 0.02]); Om_o = np.diag(1/odo_sig**2)
    odo_edges = []; poses0 = [true[0].copy()]

    def rel(a, b): return t2v(np.linalg.inv(v2t(a)) @ v2t(b))
    for k in range(n):
        z = rel(true[k], true[k+1]) + rng.normal(0, odo_sig)
        odo_edges.append((k, k+1, z, Om_o))
        poses0.append(t2v(v2t(poses0[-1]) @ v2t(z)))
    poses0 = np.array(poses0)

    # range-bearing 관측 + 랜드마크 초기화(첫 관측)
    r_sig = np.array([0.3, 0.02]); Om_z = np.diag(1/r_sig**2)
    obs = []; lms0 = np.full((L, 2), np.nan); seen = [False]*L
    for k in range(P):
        for l in range(L):
            d = lms_true[l] - true[k, :2]; rr = np.hypot(*d)
            if rr > SENSOR_RANGE:
                continue
            z = np.array([rr + rng.normal(0, r_sig[0]),
                          wrap(np.arctan2(d[1], d[0]) - true[k, 2] + rng.normal(0, r_sig[1]))])
            obs.append((k, l, z, Om_z))
            if not seen[l]:
                rx, ry, rth = poses0[k]
                lms0[l] = [rx + z[0]*np.cos(z[1]+rth), ry + z[0]*np.sin(z[1]+rth)]
                seen[l] = True
    lms0 = np.where(np.isnan(lms0), 0.0, lms0)

    poses_opt, lms_opt, hist = optimize(poses0.copy(), lms0.copy(), odo_edges, obs)

    def prmse(Pmat): return float(np.sqrt(np.mean(np.sum((Pmat[:, :2]-true[:, :2])**2, 1))))
    seen_m = np.array(seen)
    def lrmse(Lmat): return float(np.sqrt(np.mean(np.sum((Lmat[seen_m]-lms_true[seen_m])**2, 1))))

    print("=== 완전 그래프 SLAM (pose + 랜드마크 공동 최적화) ===")
    print(f"pose {P}, 랜드마크 {int(seen_m.sum())}/{L}, 오도메트리 팩터 {len(odo_edges)}, 관측 팩터 {len(obs)}")
    print(f"오도메트리 초기값 : pose RMSE={prmse(poses0):.3f} m, 지도 RMSE={lrmse(lms0):.3f} m")
    print(f"BA 최적화 후      : pose RMSE={prmse(poses_opt):.3f} m, 지도 RMSE={lrmse(lms_opt):.3f} m")
    print(f"→ pose {prmse(poses0)/max(prmse(poses_opt),1e-6):.0f}배, 지도 {lrmse(lms0)/max(lrmse(lms_opt),1e-6):.0f}배 개선")
    print(f"chi2: {hist[0]:.0f} → {hist[-1]:.3g} ({len(hist)} iters)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, Pm, Lm, title in [(ax1, poses0, lms0, f"odometry init (pose {prmse(poses0):.1f}m)"),
                              (ax2, poses_opt, lms_opt, f"joint BA (pose {prmse(poses_opt):.2f}m, map {lrmse(lms_opt):.2f}m)")]:
        ax.plot(true[:, 0], true[:, 1], "g-", lw=2.5, label="true")
        ax.plot(Pm[:, 0], Pm[:, 1], "b-", lw=1.2, label="estimate")
        ax.plot(lms_true[seen_m, 0], lms_true[seen_m, 1], "g*", ms=12)
        ax.plot(Lm[seen_m, 0], Lm[seen_m, 1], "rx", ms=8, label="landmarks")
        ax.set_aspect("equal"); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_title(title)
    fig.suptitle("Full graph SLAM: joint pose + landmark optimization (2D BA)")
    fig.tight_layout(); fig.savefig(Path("outputs") / "12_graph_slam_landmarks.png", dpi=130)
    print("\n[plot] outputs/12_graph_slam_landmarks.png")
    return prmse(poses0), prmse(poses_opt), lrmse(lms_opt)


if __name__ == "__main__":
    main()
