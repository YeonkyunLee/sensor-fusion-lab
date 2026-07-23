"""SE(3) / SO(3) Lie 군 연산 — 3D pose-graph SLAM 용.

3D 로봇 상태는 SE(3)(회전 SO(3) + 병진). 최적화는 6-DOF 접공간(se(3))에서 하고
exp 사상으로 매니폴드에 retract 한다. exp/log를 밑바닥부터 구현.
"""

from __future__ import annotations

import numpy as np


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3) + hat(phi)
    k = phi / theta
    K = hat(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def so3_log(R):
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * 0.5
    return theta / (2 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def se3_exp(xi):
    """xi = [rho(3), phi(3)] → 4x4 변환."""
    rho, phi = xi[:3], xi[3:]
    theta = np.linalg.norm(phi)
    R = so3_exp(phi)
    if theta < 1e-8:
        V = np.eye(3) + 0.5 * hat(phi)
    else:
        K = hat(phi / theta)
        V = (np.eye(3) + (1 - np.cos(theta)) / theta * K
             + (theta - np.sin(theta)) / theta * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T):
    """4x4 변환 → xi = [rho(3), phi(3)]."""
    R = T[:3, :3]
    t = T[:3, 3]
    phi = so3_log(R)
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        Vinv = np.eye(3) - 0.5 * hat(phi)
    else:
        K = hat(phi / theta)
        a = theta / 2
        Vinv = (np.eye(3) - 0.5 * hat(phi)
                + (1 - a / np.tan(a)) / theta**2 * hat(phi) @ hat(phi))
    xi = np.zeros(6)
    xi[:3] = Vinv @ t
    xi[3:] = phi
    return xi


def se3_inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti
