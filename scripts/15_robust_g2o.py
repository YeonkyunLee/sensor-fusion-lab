"""실 벤치마크(Intel) + 거짓 루프클로저 → 강건 커널 비교 (naive/Huber/DCS).

실전 SLAM은 장소 인식 오류로 거짓 루프클로저가 섞인다. 합성이 아닌 **실제 Intel g2o**에
거짓 엣지를 주입하고, 강건 백엔드가 얼마나 걸러내는지 원본(inlier) 엣지의 chi2로 잰다.

    python scripts/15_robust_g2o.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sensor_fusion.posegraph import _error_and_jacobians  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402
g2o = import_module("14_g2o_benchmark")


def robust_weight(chi2, kernel, huber_d=2.5, dcs_phi=5.0):
    if kernel == "none":
        return 1.0
    if kernel == "huber":
        r = np.sqrt(max(chi2, 1e-12))
        return 1.0 if r <= huber_d else huber_d / r
    if kernel == "dcs":
        return min(1.0, 2 * dcs_phi / (dcs_phi + chi2)) ** 2
    return 1.0


def optimize(X, E, kernel, iters=15):
    X = X.copy(); n = X.shape[0]; N = 3 * n
    for _ in range(iters):
        rows, cols, vals = [], [], []; b = np.zeros(N)
        for (i, j, z, Om) in E:
            e, A, B = _error_and_jacobians(X[i], X[j], z)
            # 오도메트리(연속 엣지)는 백본 → 항상 full weight; 루프클로저만 강건화
            w = 1.0 if abs(i - j) == 1 else robust_weight(float(e @ Om @ e), kernel)
            WOm = w * Om
            bi, bj = 3*i, 3*j
            for (M, r, c) in [(A.T@WOm@A, bi, bi), (A.T@WOm@B, bi, bj),
                              (B.T@WOm@A, bj, bi), (B.T@WOm@B, bj, bj)]:
                for a in range(3):
                    for d in range(3):
                        rows.append(r+a); cols.append(c+d); vals.append(M[a, d])
            b[bi:bi+3] += A.T@WOm@e; b[bj:bj+3] += B.T@WOm@e
        for a in range(3):
            rows.append(a); cols.append(a); vals.append(1e6)
        H = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        dx = spsolve(H, -b)
        X = X + dx.reshape(n, 3); X[:, 2] = (X[:, 2]+np.pi) % (2*np.pi) - np.pi
        if np.max(np.abs(dx)) < 1e-4:
            break
    return X


def main():
    f = Path("data_cache/intel.g2o")
    if not f.exists():
        print("intel.g2o 없음 — 표준 g2o 벤치마크 다운로드 필요"); return
    X0, E = g2o.load_g2o_se2(str(f))
    inlier_edges = list(E)

    # 거짓 루프클로저 주입: 무관한 pose 쌍에 임의 상대 측정
    rng = np.random.default_rng(0); n = X0.shape[0]; n_false = 30
    Om_f = np.diag([1/0.1**2, 1/0.1**2, 1/0.05**2])
    E_corrupt = list(E)
    for _ in range(n_false):
        i, j = rng.integers(0, n), rng.integers(0, n)
        if abs(i-j) < 20:
            continue
        z = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-np.pi, np.pi)])
        E_corrupt.append((i, j, z, Om_f))

    def inlier_chi2(X):
        return g2o.chi2_se2(X, inlier_edges)

    results = {}
    for kernel in ["none", "huber", "dcs"]:
        Xo = optimize(X0.copy(), E_corrupt, kernel)
        results[kernel] = (Xo, inlier_chi2(Xo))

    print("=== 실 Intel + 거짓 루프클로저 30개: 강건 커널 비교 ===")
    print(f"원본 엣지 {len(inlier_edges)}개 기준 inlier chi2 (낮을수록 지도 깨끗):")
    for k in ["none", "huber", "dcs"]:
        print(f"  {k:6s}: {results[k][1]:.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, k in zip(axes, ["none", "huber", "dcs"]):
        X = results[k][0]
        ax.plot(X[:, 0], X[:, 1], "-", lw=0.6, color={"none": "r", "huber": "orange", "dcs": "b"}[k])
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(f"{k}  (inlier chi2={results[k][1]:.0f})")
    fig.suptitle("Robust SLAM on real Intel g2o + 30 false loop closures")
    fig.tight_layout(); fig.savefig("outputs/15_robust_g2o.png", dpi=130)
    print("\n[plot] outputs/15_robust_g2o.png")
    return results["none"][1], results["huber"][1], results["dcs"][1]


if __name__ == "__main__":
    main()
