"""3R 매니퓰레이터 기구학 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "manip39", ROOT / "scripts" / "39_manipulator_kinematics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ik_roundtrip_reaches_reachable_target():
    """FK∘IK 왕복: 도달 가능한 목표를 DLS IK가 작은 오차로 맞춘다."""
    mod = _load()
    rng = np.random.default_rng(0)
    # 무작위 관절각으로 FK → 확실히 도달 가능한 목표 생성 → 다른 초기값에서 IK
    for _ in range(8):
        q_true = rng.uniform(-np.pi, np.pi, 3)
        target = mod.fk(q_true)[:2]
        q0 = q_true + rng.uniform(-0.4, 0.4, 3)
        q_sol, iters, _, err = mod.ik_dls(target, q0, lam=0.02, max_iters=400)
        assert err < 1e-3, f"IK 위치오차 과다: {err}"
        # FK로 되돌리면 목표와 일치
        assert np.linalg.norm(mod.fk(q_sol)[:2] - target) < 1e-3


def test_jacobian_matches_finite_difference():
    """해석 야코비안이 유한차분 야코비안과 일치."""
    mod = _load()
    rng = np.random.default_rng(1)

    def num_jac(q, eps=1e-6):
        Jn = np.zeros((3, 3))
        for i in range(3):
            dq = np.zeros(3); dq[i] = eps
            f1, f0 = mod.fk(q + dq), mod.fk(q - dq)
            Jn[:, i] = (f1 - f0) / (2 * eps)
        return Jn

    for _ in range(6):
        q = rng.uniform(-np.pi, np.pi, 3)
        assert np.max(np.abs(mod.jacobian(q) - num_jac(q))) < 1e-6


def test_dls_bounded_where_pseudo_blows_up():
    """특이점(곧게 편 자세) 근처: pseudo-inverse는 폭발, DLS는 유계로 수렴."""
    mod = _load()
    q_start = np.array([1e-3, 1e-3, 1e-3])          # 거의 특이한 자세
    target = np.array([mod.REACH - 0.02, 0.25])     # 특이 방향을 강하게 요구

    # 특이 자세에서 조작성 ≈ 0 (rank 손실)
    assert mod.manipulability(np.zeros(3)) < 1e-6

    _, _, tr_dls, dls_err = mod.ik_dls(target, q_start, lam=0.15, max_iters=200)
    _, _, tr_ps, ps_err, ps_maxstep = mod.ik_pseudo(target, q_start, max_iters=200)

    dls_maxstep = float(np.max(np.linalg.norm(np.diff(tr_dls, axis=0), axis=1)))

    # DLS는 유계(작은 스텝)로 남고 잔차도 작다
    assert dls_maxstep < 1.0, f"DLS 스텝이 유계가 아님: {dls_maxstep}"
    assert dls_err < 1e-2, f"DLS 수렴 실패: {dls_err}"
    # 순수 유사역행렬은 특이점에서 스텝이 폭발(DLS보다 최소 50배)
    assert ps_maxstep > 50 * dls_maxstep, f"pseudo가 폭발하지 않음: {ps_maxstep}"


def test_main_metrics():
    """main() 헤드라인 지표가 합리적 범위."""
    mod = _load()
    pos_err, iters, w_sol, dls_err = mod.main()
    assert pos_err < 1e-3
    assert iters < 100
    assert w_sol > 0.1            # 해가 특이점에서 멀다
    assert dls_err < 1e-2
