"""UR5 6축 기구학·동역학 코어(`sensor_fusion.ur5`) 테스트. 실행: pytest -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensor_fusion import ur5  # noqa: E402
from sensor_fusion.se3 import so3_log  # noqa: E402


def test_zero_pose_matches_published_ur5_geometry():
    """영점 자세 TCP 가 공개 DH 로부터 예상되는 값과 일치(파라미터 표 검증)."""
    p = ur5.fk(np.zeros(6))[:3, 3]
    expect = np.array([ur5.DH_A[1] + ur5.DH_A[2],                 # -0.81725
                       -(ur5.DH_D[3] + ur5.DH_D[5]),              # -0.19145
                       ur5.DH_D[0] - ur5.DH_D[4]])                # -0.00549
    assert np.allclose(p, expect, atol=1e-9), f"{p} vs {expect}"


def test_geometric_jacobian_matches_finite_differences():
    """기하 야코비안(위치 3 + 자세 3)이 수치 미분과 일치."""
    rng = np.random.default_rng(0)

    def num_jac(q, eps=1e-6):
        J = np.zeros((6, 6))
        for i in range(6):
            dq = np.zeros(6)
            dq[i] = eps
            T1, T0 = ur5.fk(q + dq), ur5.fk(q - dq)
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / (2 * eps)
            J[3:, i] = so3_log(T1[:3, :3] @ T0[:3, :3].T) / (2 * eps)
        return J

    for _ in range(5):
        q = rng.uniform(-2, 2, 6)
        assert np.max(np.abs(ur5.jacobian(q) - num_jac(q))) < 1e-7


def test_wrist_singularity_has_zero_manipulability():
    """UR5 손목 특이점(q5=0)에서 조작성이 0 으로 떨어진다."""
    q = np.array([0.3, -1.2, 1.4, -1.0, 0.0, 0.2])
    assert ur5.manipulability(q) < 1e-9
    q_ok = np.array([0.3, -1.2, 1.4, -1.0, -1.5, 0.2])
    assert ur5.manipulability(q_ok) > 1e-2


def test_6dof_ik_recovers_pose_including_orientation():
    """위치+자세 6-DOF IK 왕복: µm·arcsec 수준으로 수렴."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        q_true = np.array([0.4, -1.3, 1.3, -1.1, -1.4, 0.3]) + rng.uniform(-0.3, 0.3, 6)
        T_des = ur5.fk(q_true)
        q0 = q_true + rng.uniform(-0.25, 0.25, 6)
        q, iters, ep, er = ur5.ik_dls(T_des, q0, lam=0.01)
        assert ep < 1e-5, f"위치 잔차 {ep*1e6:.1f} µm"
        assert er < 1e-5, f"자세 잔차 {np.rad2deg(er)*3600:.1f} arcsec"
        assert iters < 100


def test_rnea_agrees_with_lagrangian_dynamics():
    """독립 구현 교차검증: RNEA 역동역학 == 라그랑주 조립(M,C,g)."""
    rng = np.random.default_rng(2)
    for _ in range(6):
        q = rng.uniform(-2, 2, 6)
        qd = rng.uniform(-1.5, 1.5, 6)
        qdd = rng.uniform(-3, 3, 6)
        tau_lag = ur5.inverse_dynamics(q, qd, qdd)
        tau_rne = ur5.rnea(q, qd, qdd)
        assert np.allclose(tau_lag, tau_rne, atol=1e-6), f"{tau_lag} vs {tau_rne}"
        assert np.allclose(ur5.mass_matrix(q), ur5.mass_matrix_rnea(q), atol=1e-9)


def test_mass_matrix_is_symmetric_positive_definite():
    rng = np.random.default_rng(3)
    for _ in range(5):
        M = ur5.mass_matrix(rng.uniform(-2, 2, 6))
        assert np.allclose(M, M.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_gravity_torque_equals_potential_gradient():
    """g(q) = ∂U/∂q 수치검증."""
    q = np.array([0.3, -1.2, 1.4, -1.0, -1.5, 0.2])

    def potential(qq):
        Ts = ur5.fk_all(qq)
        return -sum(ur5.MASS[i] * ur5.GRAVITY
                    @ (Ts[i][:3, 3] + Ts[i][:3, :3] @ ur5.COM[i]) for i in range(6))

    g_num = np.zeros(6)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = 1e-6
        g_num[i] = (potential(q + dq) - potential(q - dq)) / 2e-6
    assert np.allclose(g_num, ur5.gravity_torque(q), atol=1e-6)


def test_energy_is_conserved_without_input():
    """무토크·무마찰 적분에서 총에너지 보존 — M·C 가 서로 일관된다는 증거."""
    q = np.array([0.3, -1.0, 1.2, -0.8, -1.3, 0.4])
    qd = np.array([0.4, -0.3, 0.5, 0.2, -0.4, 0.6])
    E0 = ur5.energy(q, qd)
    dt = 1e-3
    for _ in range(200):
        def deriv(s):
            return np.concatenate(
                [s[6:], ur5.forward_dynamics_fast(s[:6], s[6:], np.zeros(6))])
        s = np.concatenate([q, qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q, qd = s[:6], s[6:]
    assert abs(ur5.energy(q, qd) - E0) / abs(E0) < 1e-6


def test_fast_forward_dynamics_matches_reference():
    """시뮬레이션용 빠른 경로가 라그랑주 기준 구현과 동일한 결과."""
    rng = np.random.default_rng(4)
    for _ in range(5):
        q = rng.uniform(-2, 2, 6)
        qd = rng.uniform(-1, 1, 6)
        tau = rng.uniform(-20, 20, 6)
        assert np.allclose(ur5.forward_dynamics(q, qd, tau),
                           ur5.forward_dynamics_fast(q, qd, tau), atol=1e-8)
