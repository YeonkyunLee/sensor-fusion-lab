"""표준 g2o 벤치마크로 pose-graph SLAM 검증 (합성 아닌 실제 데이터).

Intel(2D), parking-garage(3D) 등 SLAM 커뮤니티 표준 데이터셋(.g2o)을 파싱해 내 pose-graph
최적화기로 최적화한다. 합성 데모를 넘어 "표준 벤치마크에서 동작"을 보인다.
희소행렬(scipy.sparse)로 대규모(수천 pose) 정규방정식을 푼다.

    python scripts/14_g2o_benchmark.py --file data_cache/intel.g2o
    python scripts/14_g2o_benchmark.py --file data_cache/parking-garage.g2o --dim 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import _error_and_jacobians, t2v, v2t, wrap  # noqa: E402
from sensor_fusion.posegraph3d import _err as err3, _jac as jac3  # noqa: E402
from sensor_fusion.se3 import se3_exp  # noqa: E402


def quat2T(x, y, z, qx, qy, qz, qw):
    n = np.sqrt(qx*qx+qy*qy+qz*qz+qw*qw); qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [x, y, z]
    return T


def load_g2o_se3(path):
    verts = {}; edges = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "VERTEX_SE3:QUAT":
            verts[int(p[1])] = quat2T(*map(float, p[2:9]))
        elif p[0] == "EDGE_SE3:QUAT":
            i, j = int(p[1]), int(p[2])
            Z = quat2T(*map(float, p[3:10]))
            u = list(map(float, p[10:31]))  # 6x6 상삼각 21개
            Om = np.zeros((6, 6)); it = iter(u)
            for a in range(6):
                for c in range(a, 6):
                    Om[a, c] = Om[c, a] = next(it)
            edges.append((i, j, Z, Om))
    ids = sorted(verts); idx = {v: k for k, v in enumerate(ids)}
    X = [verts[i] for i in ids]
    E = [(idx[i], idx[j], Z, Om) for (i, j, Z, Om) in edges]
    return X, E


def chi2_se3(X, E):
    c = 0.0
    for (i, j, Z, Om) in E:
        e = err3(X[i], X[j], Z); c += float(e @ Om @ e)
    return c


def optimize_se3_sparse(X, E, iters=10):
    n = len(X); N = 6 * n
    hist = [chi2_se3(X, E)]
    for _ in range(iters):
        rows, cols, vals = [], [], []; b = np.zeros(N)
        for (i, j, Z, Om) in E:
            e, A, B = jac3(X[i], X[j], Z)
            bi, bj = 6*i, 6*j
            for (M, r, c) in [(A.T@Om@A, bi, bi), (A.T@Om@B, bi, bj),
                              (B.T@Om@A, bj, bi), (B.T@Om@B, bj, bj)]:
                for a in range(6):
                    for d in range(6):
                        rows.append(r+a); cols.append(c+d); vals.append(M[a, d])
            b[bi:bi+6] += A.T@Om@e; b[bj:bj+6] += B.T@Om@e
        for a in range(6):
            rows.append(a); cols.append(a); vals.append(1e6)
        H = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        dx = spsolve(H, -b)
        X = [X[k] @ se3_exp(dx[6*k:6*k+6]) for k in range(n)]
        hist.append(chi2_se3(X, E))
        if np.max(np.abs(dx)) < 1e-4:
            break
    return X, hist


# ---------- 2D (SE2) ----------
def load_g2o_se2(path):
    verts = {}; edges = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "VERTEX_SE2":
            verts[int(p[1])] = np.array([float(p[2]), float(p[3]), float(p[4])])
        elif p[0] == "EDGE_SE2":
            i, j = int(p[1]), int(p[2])
            z = np.array([float(p[3]), float(p[4]), float(p[5])])
            u = list(map(float, p[6:12]))
            Om = np.array([[u[0], u[1], u[2]], [u[1], u[3], u[4]], [u[2], u[4], u[5]]])
            edges.append((i, j, z, Om))
    ids = sorted(verts); idx = {v: k for k, v in enumerate(ids)}
    X = np.array([verts[i] for i in ids])
    E = [(idx[i], idx[j], z, Om) for (i, j, z, Om) in edges]
    return X, E


def chi2_se2(X, E):
    c = 0.0
    for (i, j, z, Om) in E:
        e, _, _ = _error_and_jacobians(X[i], X[j], z)
        c += float(e @ Om @ e)
    return c


def optimize_se2_sparse(X, E, iters=15):
    n = X.shape[0]; N = 3 * n
    hist = [chi2_se2(X, E)]
    for _ in range(iters):
        rows, cols, vals = [], [], []
        b = np.zeros(N)
        for (i, j, z, Om) in E:
            e, A, B = _error_and_jacobians(X[i], X[j], z)
            bi, bj = 3*i, 3*j
            for (M, r, c) in [(A.T@Om@A, bi, bi), (A.T@Om@B, bi, bj),
                              (B.T@Om@A, bj, bi), (B.T@Om@B, bj, bj)]:
                for a in range(3):
                    for d in range(3):
                        rows.append(r+a); cols.append(c+d); vals.append(M[a, d])
            b[bi:bi+3] += A.T@Om@e; b[bj:bj+3] += B.T@Om@e
        # pose 0 고정
        for a in range(3):
            rows.append(a); cols.append(a); vals.append(1e6)
        H = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        dx = spsolve(H, -b)
        X = X + dx.reshape(n, 3)
        X[:, 2] = (X[:, 2] + np.pi) % (2*np.pi) - np.pi
        hist.append(chi2_se2(X, E))
        if np.max(np.abs(dx)) < 1e-4:
            break
    return X, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data_cache/intel.g2o")
    ap.add_argument("--dim", type=int, default=2)
    args = ap.parse_args()

    if args.dim == 2:
        X0, E = load_g2o_se2(args.file)
        print(f"=== g2o 벤치마크: {Path(args.file).name} (2D) ===")
        print(f"pose {X0.shape[0]}, edge {len(E)}")
        c0 = chi2_se2(X0, E)
        Xopt, hist = optimize_se2_sparse(X0.copy(), E)
        print(f"chi2: {c0:.1f} → {hist[-1]:.1f}  ({(1-hist[-1]/c0)*100:.1f}% 감소, {len(hist)-1} iters)")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
        a1.plot(X0[:, 0], X0[:, 1], "r-", lw=0.6); a1.set_title(f"initial (chi2={c0:.0f})")
        a2.plot(Xopt[:, 0], Xopt[:, 1], "b-", lw=0.6); a2.set_title(f"optimized (chi2={hist[-1]:.0f})")
        for a in (a1, a2):
            a.set_aspect("equal"); a.grid(alpha=0.3)
        fig.suptitle(f"g2o benchmark: {Path(args.file).name} — pose-graph optimization")
        out = f"outputs/14_g2o_{Path(args.file).stem}.png"
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"[plot] {out}")
        return c0, hist[-1]
    else:
        X0, E = load_g2o_se3(args.file)
        print(f"=== g2o 벤치마크: {Path(args.file).name} (3D SE3) ===")
        print(f"pose {len(X0)}, edge {len(E)}")
        c0 = chi2_se3(X0, E)
        Xopt, hist = optimize_se3_sparse([T.copy() for T in X0], E)
        print(f"chi2: {c0:.1f} → {hist[-1]:.1f}  ({(1-hist[-1]/c0)*100:.1f}% 감소, {len(hist)-1} iters)")
        P0 = np.array([T[:3, 3] for T in X0]); Po = np.array([T[:3, 3] for T in Xopt])
        fig = plt.figure(figsize=(13, 6))
        for k, (P, title) in enumerate([(P0, f"initial (chi2={c0:.0f})"), (Po, f"optimized (chi2={hist[-1]:.0f})")]):
            ax = fig.add_subplot(1, 2, k+1, projection="3d")
            ax.plot(P[:, 0], P[:, 1], P[:, 2], "-", lw=0.5, color=("r" if k == 0 else "b"))
            ax.set_title(title)
        fig.suptitle(f"g2o benchmark: {Path(args.file).name} — 3D SE(3) pose-graph optimization")
        out = f"outputs/14_g2o_{Path(args.file).stem}.png"
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"[plot] {out}")
        return c0, hist[-1]


if __name__ == "__main__":
    main()
