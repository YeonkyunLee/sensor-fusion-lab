"""2링크 팔 동역학 + 계산토크 제어 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "arm40", ROOT / "scripts" / "40_manipulator_dynamics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_computed_torque_beats_pd_and_is_accurate():
    mod = _load()
    pd_rmse, gc_rmse, ct_rmse = mod.main()

    # (i) 계산토크가 PD-only 보다 확실히 우수 (한 자릿수 이상 개선)
    assert ct_rmse < pd_rmse, f"computed-torque({ct_rmse}) should beat PD({pd_rmse})"
    assert ct_rmse < 0.1 * pd_rmse, "computed-torque should be far tighter than PD"

    # (ii) 계산토크 RMSE 가 작은 절대 임계 이하
    assert ct_rmse < 1e-3, f"computed-torque RMSE too high: {ct_rmse}"

    # 중력보상은 PD 와 계산토크 사이(단조 개선)
    assert ct_rmse < gc_rmse < pd_rmse


def test_mass_matrix_symmetric_positive_definite():
    mod = _load()
    p = mod.ArmParams()
    for q in ([0.0, 0.0], [0.3, 0.7], [-0.5, 1.2], [1.0, -0.8]):
        M = mod.mass_matrix(np.array(q), p)
        assert np.allclose(M, M.T), f"M not symmetric at q={q}"
        assert np.all(np.linalg.eigvalsh(M) > 0), f"M not positive-definite at q={q}"


def test_forward_dynamics_consistent_with_inverse():
    """순동역학과 역동역학의 왕복 일관성: τ→q̈→τ 재구성이 일치해야 한다."""
    mod = _load()
    p = mod.ArmParams()
    q = np.array([0.4, -0.6])
    qd = np.array([0.7, -0.3])
    tau = np.array([2.5, -1.1])
    qdd = mod.forward_dynamics(q, qd, tau, p)
    tau_rec = mod.mass_matrix(q, p) @ qdd \
        + mod.coriolis_matrix(q, qd, p) @ qd + mod.gravity_vector(q, p)
    assert np.allclose(tau, tau_rec, atol=1e-9)
