"""휨 보상 폐루프(exp 54) 테스트. 외부 데이터 불필요 — 전부 모델 기반.

실행: pytest -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "needle54", ROOT / "scripts" / "54_closed_loop_needle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 코어: 제어 법칙과 그 특수해
# --------------------------------------------------------------------------- #
def test_flip_law_reproduces_the_analytic_optimum_of_exp48():
    """균질 조직에서 F(d)=F(L)/2 의 해는 정확히 L(1−1/√2) — exp 48 의 해석해와 같아야 한다.

    이게 이 실험 전체의 교차검증 앵커다(제어 법칙이 앞 실험의 특수해를 포함하는가)."""
    m = _load()
    expected = (1.0 - 1.0 / np.sqrt(2)) * m.L_INS
    for k in (0.3, 0.81, 2.0, 5.0):
        assert abs(m.plan_flip(k, k) - expected) < 1e-9, f"κ={k} 에서 어긋남"


def test_optimal_flip_depends_on_the_layer_ratio():
    """층이 생기면 최적 시점이 조직에 의존한다 — 이 실험의 전제.

    r>1(층 2 가 더 휨)이면 모멘트가 빨리 쌓이므로 최적 시점이 **앞으로** 와야 한다."""
    m = _load()
    ds = [m.plan_flip(m.KAPPA1, m.KAPPA1 * r) for r in (0.4, 1.0, 2.5)]
    assert ds[0] < ds[1] < ds[2], f"단조가 아니다: {ds}"
    assert ds[2] - ds[0] > 0.1 * m.L_INS, "의존성이 너무 약하면 실험이 성립하지 않는다"


def test_optimal_flip_nulls_the_tip_deviation():
    """계획 법칙이 실제로(비선형 적분에서도) 팁 편차를 없애야 한다."""
    m = _load()
    for r in (0.4, 1.0, 1.8, 2.5):
        k2 = m.KAPPA1 * r
        e = m.tip_error(m.KAPPA1, k2, float(m.plan_flip(m.KAPPA1, k2)))
        assert e < 0.05e-3, f"r={r} 에서 잔차 {e*1e3:.3f} mm — 소각 모델 오차보다 크다"


def test_open_loop_fails_when_the_ratio_departs_from_one():
    """공칭 29.3% 는 r=1 에서만 맞다."""
    m = _load()
    assert m.tip_error(m.KAPPA1, m.KAPPA1, float(m.D_NOMINAL)) < 0.05e-3
    assert m.tip_error(m.KAPPA1, m.KAPPA1 * 2.5, float(m.D_NOMINAL)) > 1.0e-3


# --------------------------------------------------------------------------- #
# 코어: 관측성 — 왜 측정이 늦게 오는가
# --------------------------------------------------------------------------- #
def test_position_carries_curvature_information_only_quadratically():
    """팁 **위치**는 곡률의 이중적분이라 κ₂ 계수가 (S−s_b)²/2 로 자란다."""
    m = _load()
    s_b = m.S_BOUND
    for d in (1e-3, 3e-3, 8e-3):
        S = s_b + d
        assert abs(m.design_row(S, s_b)[1] - d ** 2 / 2) < 1e-15
    assert m.design_row(s_b - 1e-3, s_b)[1] == 0.0, "경계 전에는 정보가 없어야 한다"


def test_orientation_carries_it_linearly_and_therefore_earlier():
    """팁 **방향**은 1차 적분이라 (S−s_b) 로 선형 — 위치보다 훨씬 일찍 정보가 생긴다."""
    m = _load()
    s_b = m.S_BOUND
    d = 5e-3
    a_pos = m.design_row(s_b + d, s_b)[1]
    a_ori = m.design_row_theta(s_b + d, s_b)[1]
    assert abs(a_ori - d) < 1e-15
    # 같은 상대 잡음 기준으로 방향 쪽 정보가 압도적이어야 한다
    info_pos = (a_pos / m.TIP_SIGMA) ** 2
    info_ori = (a_ori / m.THETA_SIGMA) ** 2
    assert info_ori > 100 * info_pos, f"{info_ori:.3g} vs {info_pos:.3g}"


def test_the_estimate_is_worse_than_the_prior_at_the_decision_depth():
    """핵심 전제: 결정해야 할 순간에 κ̂₂ 의 불확실도가 사전분포보다 크다."""
    m = _load()
    S = np.arange(m.MEAS_STEP, m.D_NOMINAL + 1e-12, m.MEAS_STEP)
    A = np.stack([m.design_row(s) for s in S])
    sig2 = float(np.sqrt(m.TIP_SIGMA ** 2 * np.linalg.inv(A.T @ A)[1, 1]))
    prior_sd = m.KAPPA1 * m.R_PRIOR_SD
    assert sig2 > 10 * prior_sd, \
        f"σ(κ̂₂)={sig2:.2f} vs 사전 {prior_sd:.2f} — 우도가 이기면 이 실험의 전제가 깨진다"


def test_duty_saturates_to_the_flip_policy_when_it_is_late():
    """duty 는 flip 의 **상위집합**이다: 명령이 포화하면 u=−1 이 되어 정확히 flip 과 같아진다.

    그래서 duty 는 flip 보다 나빠질 수 없고, 늦었을 때 이득도 없다 — 이득은 '다르게 조작'이
    아니라 '**다시** 조작'에서 나온다는 것을 이 테스트가 고정한다."""
    m = _load()
    k2 = m.KAPPA1 * 2.5                                  # 이미 뒤처진 환자
    for d in (30e-3, 40e-3):
        u = m.duty_schedule(None, m.KAPPA1, k2, (d,), known=True)
        assert np.allclose(u[np.linspace(0, m.L_INS, len(u)) >= d], -1.0), "포화해야 한다"
        assert abs(m.tip_error(m.KAPPA1, k2, u) - m.tip_error(m.KAPPA1, k2, d)) < 1e-9


def test_repeated_replanning_is_where_duty_wins():
    """이득의 출처: 한 번 더 고칠 수 있다는 것. 재계획 1회면 flip 과 같고, 늘리면 좋아진다."""
    m = _load()
    rng = np.random.default_rng(0)
    rs = rng.uniform(*m.R_RANGE, 60)
    def p90(deps):
        e = [m.tip_error(m.KAPPA1, m.KAPPA1 * r,
                         m.duty_schedule(np.random.default_rng([1, i]), m.KAPPA1,
                                         m.KAPPA1 * r, deps))
             for i, r in enumerate(rs)]
        return float(np.percentile(e, 90))
    one, four = p90((22e-3,)), p90((22e-3, 32e-3, 42e-3, 52e-3))
    assert four < 0.9 * one, f"재계획 4회 {four*1e3:.2f} vs 1회 {one*1e3:.2f} mm"


# --------------------------------------------------------------------------- #
# 전체 실험
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def result():
    m = _load()
    return m, m.main(quick=True)


def test_measurement_contributes_almost_nothing_with_one_shot_flip(result):
    """핵심 결과 1 — 절제: flip 정책에서는 측정이 사실상 기여하지 않는다.

    측정 0회의 '사전지식만' 정책이 MAP 와 같은 수준이면, '폐루프가 좋아졌다'의 실체는
    정보가 아니라 더 맞는 기본값이다."""
    _, res = result
    pri = res["prior_only_p90"]
    mp = res["C"]["map"]["p90"][res["best"]["map"]]
    assert abs(mp - pri) < 0.1e-3, \
        f"사전지식만 {pri*1e3:.2f} vs MAP {mp*1e3:.2f} mm — 차이가 크면 절제 결론이 바뀐다"
    assert res["open_p90"] > 1.5 * pri, "열린 루프 대비 개선 자체는 있어야 한다"


def test_orientation_helps_but_does_not_rescue_the_one_shot_policy(result):
    """5-DOF 센서는 방향까지 주지만, flip 한 번으로는 간극을 거의 못 메운다."""
    _, res = result
    p5, pri, orc = res["map5_p90"], res["prior_only_p90"], res["oracle_p90"]
    assert p5 <= pri + 1e-9, "방향을 더 주고도 나빠지면 안 된다"
    assert (pri - p5) < 0.3 * (pri - orc), \
        f"방향만으로 간극의 {(pri-p5)/(pri-orc)*100:.0f}% 를 메웠다 — 결론이 달라진다"


def test_duty_cycling_beats_the_flip_policy_with_the_same_sensor(result):
    """핵심 결과 2 — 병목은 작동이었다. 센서·추정기를 그대로 두고 작동만 바꾼다."""
    _, res = result
    best_duty = min(v[1] for v in res["duty"].values())
    assert best_duty < 0.85 * res["map5_p90"], \
        f"duty {best_duty*1e3:.2f} vs flip {res['map5_p90']*1e3:.2f} mm"
    assert best_duty < res["open_p90"] * 0.6


def test_more_replans_help(result):
    """늦게 도착한 정보가 실제로 쓰인다 — 재계획을 늘릴수록 좋아져야 한다."""
    _, res = result
    p90s = [v[1] for v in res["duty"].values()]
    assert p90s[-1] < p90s[0], f"재계획 증가가 도움이 안 된다: {p90s}"


def test_estimation_is_no_longer_the_bottleneck_under_duty(result):
    """핵심 결과 3 — 병목이 또 옮겨 간다. 참 κ 를 줘도 duty 잔차가 거의 그대로다."""
    _, res = result
    best_duty = min(v[1] for v in res["duty"].values())
    known = res["duty_known_p90"]
    assert known > 0.7 * best_duty, \
        f"참 κ 로 {known*1e3:.2f} mm 까지 내려가면 추정이 여전히 병목이라는 뜻"


def test_sensor_quality_barely_moves_the_duty_result(result):
    """그래서 센서를 좋게 사는 것이 더는 답이 아니다(잡음 10배 차이가 거의 무의미)."""
    _, res = result
    g = res["D_grid"]
    assert abs(g[0, 0] - g[-1, 0]) < 0.1e-3, \
        f"σ 10배 차이가 {abs(g[0,0]-g[-1,0])*1e3:.2f} mm — 크면 결론이 바뀐다"
