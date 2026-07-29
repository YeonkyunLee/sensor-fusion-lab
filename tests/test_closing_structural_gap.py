"""구조 간극 닫기(exp 46) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "gap46", ROOT / "scripts" / "46_closing_structural_gap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_extended_regressor_reduces_to_the_old_one():
    """확장 회귀자의 앞 9열은 exp 43 회귀자와 동일해야 한다(호환성)."""
    m = _load()
    rng = np.random.default_rng(0)
    for _ in range(5):
        q, qd, qdd = (rng.uniform(-2, 2, 2), rng.uniform(-2, 2, 2),
                      rng.uniform(-3, 3, 2))
        Y_ext = m.regressor_ext(q, qd, qdd, vs=0.05)
        assert np.allclose(Y_ext[:, :9], m.s2r.regressor(q, qd, qdd))


def test_extended_regressor_identity_holds():
    """확장 항등식 Y(vs)·π ≡ 역동역학(확장 마찰 포함)."""
    m = _load()
    rng = np.random.default_rng(1)
    pi = np.concatenate([m.s2r.PI_TRUE, [1.44, 0.72]])
    for _ in range(5):
        q, qd, qdd = (rng.uniform(-2, 2, 2), rng.uniform(-1.5, 1.5, 2),
                      rng.uniform(-3, 3, 2))
        lhs = m.regressor_ext(q, qd, qdd, 0.05) @ pi
        rhs = m.inverse_dynamics_ext(pi, q, qd, qdd, 0.05)
        assert np.allclose(lhs, rhs, atol=1e-9)


def test_lowspeed_excitation_actually_visits_the_stiction_regime():
    """저속 여기 궤적은 |q̇| < vs 구간에 실제로 머문다(고속 여기는 거의 안 머문다)."""
    m = _load()
    _, Qd_slow, _ = m.excitation_lowspeed(seed=1)
    _, Qd_fast, _ = m.s2r.excitation_trajectory(seed=1)
    frac_slow = float(np.mean(np.abs(Qd_slow) < m.s2r.V_STRIBECK))
    frac_fast = float(np.mean(np.abs(Qd_fast) < m.s2r.V_STRIBECK))
    assert frac_slow > 0.3, f"저속 여기가 스틱션 영역에 머물지 않음: {frac_slow:.2f}"
    assert frac_slow > 5 * frac_fast, f"{frac_slow:.2f} vs {frac_fast:.2f}"


def test_grid_search_recovers_the_nonlinear_parameter():
    """separable LS: vs 격자 탐색의 잔차 최소점이 참 vs 를 찾는다."""
    m = _load()
    logs = []
    pi9 = m.s2r.PI_NOMINAL.copy()
    for it in (1, 2):
        Qe, Qde, Qdde = m.s2r.excitation_trajectory(seed=it)
        *_, meas = m.s2r.rollout(m.s2r.PI_TRUE, pi9, Qe, Qde, Qdde,
                                 rng=np.random.default_rng(it), log=True,
                                 stribeck=True)
        logs.append(meas)
        Qs, Qds, Qdds = m.excitation_lowspeed(seed=it)
        *_, meas_s = m.s2r.rollout(m.s2r.PI_TRUE, pi9, Qs, Qds, Qdds,
                                   rng=np.random.default_rng(100 + it), log=True,
                                   stribeck=True)
        logs.append(meas_s)

    (pi11, vs_hat, _, _), curve = m.identify_ext_search(logs)
    assert abs(vs_hat - m.s2r.V_STRIBECK) < 0.02, f"vs 추정 실패: {vs_hat}"
    fs_true = m.s2r.FC_TRUE * (m.s2r.FS_RATIO - 1.0)
    assert np.allclose(pi11[9:11], fs_true, rtol=0.35), f"{pi11[9:11]} vs {fs_true}"


def test_structure_alone_is_not_enough_but_structure_plus_data_closes_the_gap():
    """핵심 결론: 구조 확장만으로는 부족하고, 저속 데이터가 함께 와야 정체가 풀린다."""
    m = _load()
    res = m.main()

    plateau = res["base_hist"][-1]
    assert plateau > 1e-4, "기준선이 정체하지 않으면 실험 전제가 깨진다"
    # 구조만 확장: 큰 개선이 없다
    assert res["err_fast"] > plateau / 3, f"구조만으로 닫혀버림: {res['err_fast']*1e3:.3f} mm"
    # 구조 + 저속 데이터: 한 자릿수 이상 개선
    assert res["err_ext"] < plateau / 10, f"확장이 정체를 풀지 못함: {res['err_ext']*1e3:.3f} mm"
    assert abs(res["vs_hat"] - m.s2r.V_STRIBECK) < 0.02


def test_overparameterized_model_costs_something_on_a_clean_plant():
    """정직한 반대편: 구조 간극이 없으면 잉여 파라미터는 공짜가 아니다."""
    m = _load()
    res = m.main()
    assert res["err9_clean"] < 1e-5
    # 과잉 모델이 더 낫지는 않다(같거나 나쁘다)
    assert res["err11_clean"] >= res["err9_clean"]
