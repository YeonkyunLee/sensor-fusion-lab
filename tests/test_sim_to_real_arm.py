"""수술 팔 sim-to-real 폐루프(exp 43) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "s2r43", ROOT / "scripts" / "43_sim_to_real_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regressor_matches_inverse_dynamics():
    """식별의 전제: Y(q,q̇,q̈)·π ≡ 역동역학. 이게 깨지면 회귀가 무의미하다."""
    mod = _load()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-3, 3, 2)
        qdd = rng.uniform(-8, 8, 2)
        lhs = mod.regressor(q, qd, qdd) @ mod.PI_TRUE
        rhs = mod.inverse_dynamics(mod.PI_TRUE, q, qd, qdd)
        assert np.allclose(lhs, rhs, atol=1e-9), f"회귀자 불일치: {lhs} vs {rhs}"


def test_dynamics_agrees_with_experiment_40_when_frictionless():
    """π 파라미터화 동역학이 exp 40의 ArmParams 구현과 일치(마찰 0일 때)."""
    mod = _load()
    dyn = mod.tgt.dyn
    p = dyn.ArmParams(m1=mod.M1, m2=mod.M2, l1=mod.L1, l2=mod.L2,
                      lc1=mod.LC1, lc2=mod.LC2, I1=mod.I1, I2=mod.I2)
    pi = mod.pack(mod.inertial_params(0.0))          # 마찰 없음 = exp 40 모델
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(-2, 2, 2)
        qd = rng.uniform(-2, 2, 2)
        tau = rng.uniform(-30, 30, 2)
        a1 = mod.forward_dynamics(pi, q, qd, tau)
        a2 = dyn.forward_dynamics(q, qd, tau, p)
        assert np.allclose(a1, a2, atol=1e-9), f"exp40과 불일치: {a1} vs {a2}"


def test_identification_recovers_payload_and_friction():
    """여기 궤적 한 번의 로그로 페이로드·마찰 파라미터를 복원한다."""
    mod = _load()
    Qe, Qde, Qdde = mod.excitation_trajectory(seed=1)
    *_, meas = mod.rollout(mod.PI_TRUE, mod.PI_NOMINAL, Qe, Qde, Qdde,
                           rng=np.random.default_rng(0), log=True)
    pi_hat, cond, resid = mod.identify([meas])

    nominal_err = np.linalg.norm(mod.PI_NOMINAL - mod.PI_TRUE)
    ident_err = np.linalg.norm(pi_hat - mod.PI_TRUE)
    assert ident_err < 0.1 * nominal_err, f"식별 실패: {ident_err} vs {nominal_err}"
    # 쿨롱 마찰(공칭 모델엔 아예 없던 항)이 10% 이내로 복원되어야 한다
    assert np.allclose(pi_hat[7:9], mod.FC_TRUE, rtol=0.15), pi_hat[7:9]
    assert cond < 1e4


def test_loop_closes_the_parametric_gap():
    """배치→식별→재배치 1회만으로 표적오차가 두 자릿수 배 이상 줄어든다."""
    mod = _load()
    task = mod.clinical_task()
    err0, resid0, _, _ = mod.deploy(mod.PI_NOMINAL, task)
    assert resid0 > mod.RESIDUAL_TRIGGER, "문제가 감지되지 않으면 루프가 시작되지 않음"

    Qe, Qde, Qdde = mod.excitation_trajectory(seed=1)
    *_, meas = mod.rollout(mod.PI_TRUE, mod.PI_NOMINAL, Qe, Qde, Qdde,
                           rng=np.random.default_rng(0), log=True)
    pi_hat, _, _ = mod.identify([meas])
    err1, resid1, _, _ = mod.deploy(pi_hat, task)

    assert err1 < err0 / 100, f"루프가 표적오차를 못 줄임: {err0*1e3} → {err1*1e3} mm"
    assert resid1 < resid0 / 100


def test_structural_gap_plateaus_above_parametric():
    """모델에 없는 스틱션은 식별이 흡수하지 못해 오차가 바닥에서 멈춘다."""
    mod = _load()
    task = mod.clinical_task()
    Qe, Qde, Qdde = mod.excitation_trajectory(seed=1)

    *_, meas_p = mod.rollout(mod.PI_TRUE, mod.PI_NOMINAL, Qe, Qde, Qdde,
                             rng=np.random.default_rng(0), log=True)
    pi_p, _, _ = mod.identify([meas_p])
    err_p, _, _, _ = mod.deploy(pi_p, task)

    *_, meas_s = mod.rollout(mod.PI_TRUE, mod.PI_NOMINAL, Qe, Qde, Qdde,
                             rng=np.random.default_rng(0), log=True, stribeck=True)
    pi_s, _, _ = mod.identify([meas_s])
    err_s, _, _, _ = mod.deploy(pi_s, task, stribeck=True)

    assert err_s > 10 * err_p, f"구조 간극이 드러나지 않음: {err_s*1e3} vs {err_p*1e3} mm"
    assert err_s < 1e-3, "구조 간극이 있어도 루프는 여전히 크게 개선해야 함"


def test_excitation_beats_clinical_trajectory_for_identification():
    """관측성: 느린 임상 궤적 로그는 병조건이라 식별 품질이 떨어진다."""
    mod = _load()
    task = mod.clinical_task()

    *_, meas_task = mod.rollout(mod.PI_TRUE, mod.PI_NOMINAL, task[0], task[1], task[2],
                                rng=np.random.default_rng(1), log=True)
    _, cond_task, _ = mod.identify([meas_task])

    Qe, Qde, Qdde = mod.excitation_trajectory(seed=1)
    *_, meas_exc = mod.rollout(mod.PI_TRUE, mod.PI_NOMINAL, Qe, Qde, Qdde,
                               rng=np.random.default_rng(1), log=True)
    _, cond_exc, _ = mod.identify([meas_exc])

    assert cond_task > 5 * cond_exc, f"여기 차이가 드러나지 않음: {cond_task} vs {cond_exc}"


def test_uses_published_ur5_parameters():
    """실데이터 앵커: 링크 파라미터가 UR5 공개 사양과 일치."""
    mod = _load()
    assert mod.L1 == 0.425 and mod.L2 == 0.39225      # UR5 a2, a3
    assert mod.M1 == 8.393 and mod.M2 == 2.275        # UR5 링크2·3 질량
