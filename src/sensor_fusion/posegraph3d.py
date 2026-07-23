"""3D SE(3) pose-graph SLAM (Gauss-Newton, 매니폴드 최적화).

각 pose는 4x4 SE(3) 변환. 엣지 오차 e = log(Zij^-1 · Xi^-1 · Xj) ∈ R^6.
야코비안은 우측 perturbation 수치미분으로 구하고, 갱신은 Xi ← Xi·exp(dξ)로 retract.
"""

from __future__ import annotations

import numpy as np

from .se3 import se3_exp, se3_inv, se3_log


class Edge3D:
    __slots__ = ("i", "j", "Z", "Om")

    def __init__(self, i, j, Z, Om):
        self.i, self.j = i, j
        self.Z = np.asarray(Z, float)      # 4x4 상대 pose 측정
        self.Om = np.asarray(Om, float)    # 6x6 정보행렬


def _err(Xi, Xj, Z):
    return se3_log(se3_inv(Z) @ se3_inv(Xi) @ Xj)


def _jac(Xi, Xj, Z, eps=1e-6):
    e0 = _err(Xi, Xj, Z)
    A = np.zeros((6, 6)); B = np.zeros((6, 6))
    for k in range(6):
        d = np.zeros(6); d[k] = eps
        A[:, k] = (_err(Xi @ se3_exp(d), Xj, Z) - e0) / eps
        B[:, k] = (_err(Xi, Xj @ se3_exp(d), Z) - e0) / eps
    return e0, A, B


def optimize_se3(poses, edges, iters=20, tol=1e-5):
    """poses: list[4x4], edges: list[Edge3D]. pose 0 고정. (최적화된 poses, chi2 이력)."""
    X = [np.array(P, float) for P in poses]
    n = len(X)
    N = 6 * n
    hist = []
    for _ in range(iters):
        H = np.zeros((N, N)); b = np.zeros(N); chi = 0.0
        for e in edges:
            err, A, B = _jac(X[e.i], X[e.j], e.Z)
            chi += float(err @ e.Om @ err)
            i, j = 6 * e.i, 6 * e.j
            H[i:i+6, i:i+6] += A.T @ e.Om @ A
            H[i:i+6, j:j+6] += A.T @ e.Om @ B
            H[j:j+6, i:i+6] += B.T @ e.Om @ A
            H[j:j+6, j:j+6] += B.T @ e.Om @ B
            b[i:i+6] += A.T @ e.Om @ err
            b[j:j+6] += B.T @ e.Om @ err
        H[:6, :6] += np.eye(6) * 1e6            # pose 0 고정(게이지)
        dx = np.linalg.solve(H + 1e-9 * np.eye(N), -b)
        for k in range(n):
            X[k] = X[k] @ se3_exp(dx[6*k:6*k+6])   # 우측 retraction
        hist.append(chi)
        if np.max(np.abs(dx)) < tol:
            break
    return X, hist
