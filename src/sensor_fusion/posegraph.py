"""2D pose-graph SLAM (SE(2) Gauss-Newton 최적화).

노드=로봇 pose(x,y,θ), 엣지=상대 pose 제약(오도메트리·루프클로저). 모든 pose를
동시에 최적화해 제약 오차 제곱합을 최소화한다. EKF와 달리 **과거 전체를 재선형화**하므로
루프 클로저 시 궤적 전체가 정렬된다.

참고: Grisetti et al., "A tutorial on graph-based SLAM".
"""

from __future__ import annotations

import numpy as np


def wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def v2t(v):
    """(x,y,θ) → 3x3 동차변환."""
    c, s = np.cos(v[2]), np.sin(v[2])
    return np.array([[c, -s, v[0]], [s, c, v[1]], [0, 0, 1.0]])


def t2v(T):
    """3x3 동차변환 → (x,y,θ)."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


class Edge:
    """i→j 상대 pose 측정 z(3,)와 정보행렬 Ω(3x3)."""

    __slots__ = ("i", "j", "z", "omega")

    def __init__(self, i, j, z, omega):
        self.i, self.j = i, j
        self.z = np.asarray(z, float)
        self.omega = np.asarray(omega, float)


def _error_and_jacobians(xi, xj, z):
    """엣지 오차 e(3,)와 야코비안 A=de/dxi, B=de/dxj (각 3x3)."""
    ti, tj, tz = xi[:2], xj[:2], z[:2]
    c, s = np.cos(xi[2]), np.sin(xi[2])
    RiT = np.array([[c, s], [-s, c]])
    dRiT = np.array([[-s, c], [-c, -s]])
    cz, sz = np.cos(z[2]), np.sin(z[2])
    RzT = np.array([[cz, sz], [-sz, cz]])

    e_t = RzT @ (RiT @ (tj - ti) - tz)
    e = np.array([e_t[0], e_t[1], wrap(xj[2] - xi[2] - z[2])])

    A = np.zeros((3, 3))
    A[:2, :2] = -RzT @ RiT
    A[:2, 2] = RzT @ (dRiT @ (tj - ti))
    A[2, 2] = -1.0
    B = np.zeros((3, 3))
    B[:2, :2] = RzT @ RiT
    B[2, 2] = 1.0
    return e, A, B


def optimize(x0: np.ndarray, edges: list[Edge], iters: int = 20, tol: float = 1e-4):
    """pose 벡터 x0 (n,3)를 Gauss-Newton으로 최적화. pose 0을 앵커로 고정."""
    x = np.asarray(x0, float).copy()
    n = x.shape[0]
    N = 3 * n
    history = []
    for _ in range(iters):
        H = np.zeros((N, N))
        b = np.zeros(N)
        chi2 = 0.0
        for e in edges:
            err, A, B = _error_and_jacobians(x[e.i], x[e.j], e.z)
            chi2 += float(err @ e.omega @ err)
            i, j = 3 * e.i, 3 * e.j
            H[i:i+3, i:i+3] += A.T @ e.omega @ A
            H[i:i+3, j:j+3] += A.T @ e.omega @ B
            H[j:j+3, i:i+3] += B.T @ e.omega @ A
            H[j:j+3, j:j+3] += B.T @ e.omega @ B
            b[i:i+3] += A.T @ e.omega @ err
            b[j:j+3] += B.T @ e.omega @ err
        H[:3, :3] += np.eye(3) * 1e6   # pose 0 고정(게이지 앵커)
        dx = np.linalg.solve(H, -b)
        x += dx.reshape(n, 3)
        x[:, 2] = np.array([wrap(a) for a in x[:, 2]])
        history.append(chi2)
        if np.max(np.abs(dx)) < tol:
            break
    return x, history
