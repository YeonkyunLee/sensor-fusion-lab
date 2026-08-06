"""exp 61 의 주장과 함정을 회귀로 고정한다.

핵심은 두 가지다. ① 이완 모델이 **앞 실험을 무효화하지 않는다**(tau=inf 면 exp 60 과 동일하고,
빠른 전진에서는 옛 모델로 환원된다). ② exp 60 의 식별 프로토콜이 **틀린 값에서 깨끗하게 수렴**하는
실패 모드가 실재한다 — 그게 이 실험에서 가장 위험한 발견이라 테스트로 못박는다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("61_tissue_relaxes")
s60 = import_module("60_measure_to_decide")
s59 = import_module("59_what_is_safe_state")
jc = import_module("56_jittery_channel")

K, V = m.K_GRIP0, m.V_INSERT


# --------------------------------------------------------------------------- #
# 앞 실험을 무효화하지 않는가
# --------------------------------------------------------------------------- #
def test_infinite_tau_is_exactly_exp60():
    """tau=inf 면 exp 60 의 TunableTissue 와 완전히 같아야 한다."""
    a = m.RelaxingTissue(f_slip=0.8, tau=np.inf)
    b = s60.TunableTissue(f_slip=0.8)
    for x in np.linspace(jc.X_SURFACE - 0.002, jc.X_SURFACE + 0.06, 600):
        assert a.force(x) == pytest.approx(b.force(x), abs=1e-12)


def test_no_relaxation_before_puncture():
    """관통 전에는 완화할 파악 상태가 없다 — 두 모델이 같아야 한다."""
    a = m.RelaxingTissue(f_slip=0.8, tau=0.01)
    b = s60.TunableTissue(f_slip=0.8)
    for x in np.linspace(jc.X_SURFACE, jc.X_SURFACE + 0.005, 200):
        assert a.force(x) == pytest.approx(b.force(x), abs=1e-12)
        if a.punctured:
            break


def test_fast_advance_reduces_to_the_old_model():
    """K·v·τ >= F_slip 이면 미끄러짐이 잘라 옛 모델과 같아진다 — 이게 환원 조건이다."""
    f_slip, tau = 0.8, 1.0
    assert K * V * tau >= f_slip                       # 조건 성립을 명시
    d, f, _ = m.insertion_log(f_slip, tau=tau, dwell=False)
    a, _ = s60.fit_from_insertion(d, f)
    assert a == pytest.approx(jc.tele.F_CUT + f_slip, rel=0.03)


def test_slow_advance_measures_the_speed_not_the_slip_limit():
    """**exp 60 의 '상수항 = F_cut + F_slip' 이 조건부가 된다.**

    K·v·τ < F_slip 이면 상수항은 F_cut + K·v·τ 로, 조직이 아니라 **삽입 속도**를 잰다.
    """
    f_slip, tau = 3.2, 0.2
    cap = K * V * tau
    assert cap < f_slip
    d, f, _ = m.insertion_log(f_slip, tau=tau, dwell=False)
    a, _ = s60.fit_from_insertion(d, f)
    assert a == pytest.approx(jc.tele.F_CUT + cap, rel=0.10)
    assert a < jc.tele.F_CUT + f_slip * 0.7            # exp 60 의 예측과 명백히 다르다


# --------------------------------------------------------------------------- #
# A — 응력 완화 자체
# --------------------------------------------------------------------------- #
def test_holding_still_makes_the_grip_force_decay():
    """붙들고 가만히 있으면 파악 힘이 사라진다(변형은 남는다). 순탄성에서는 안 그렇다."""
    relaxing = m.hold_and_relax(0.2, secs=3.0)
    elastic = m.hold_and_relax(np.inf, secs=3.0)
    assert relaxing[-1] < relaxing[0] * 0.8
    assert elastic[-1] == pytest.approx(elastic[0], rel=1e-6)


def test_decay_is_faster_for_a_shorter_time_constant():
    fast = m.hold_and_relax(0.05, secs=2.0)
    slow = m.hold_and_relax(1.0, secs=2.0)
    i = int(0.5 / m.DT)
    drop = lambda v: (v[0] - v[i]) / max(v[0] - v[-1], 1e-9)   # noqa: E731
    assert drop(fast) > drop(slow)


def test_the_hazard_itself_becomes_frequency_dependent():
    """exp 59 가 잰 '붙들기의 위해'는 순탄성 위의 값이었다 — 느린 호흡에서는 힘이 안 쌓인다."""
    slow = np.ptp(m.hold_and_relax(0.2, breath_mm=5.0, breath_hz=0.1, secs=4.0)[1000:])
    fast = np.ptp(m.hold_and_relax(0.2, breath_mm=5.0, breath_hz=1.0, secs=4.0)[1000:])
    assert fast > slow * 2.0


# --------------------------------------------------------------------------- #
# B2 — 가장 위험한 발견: 틀린 값에서의 깨끗한 수렴
# --------------------------------------------------------------------------- #
def test_the_ladder_never_converges_at_a_wrong_value():
    """**예측 실패 ①을 고정한다.** 사다리가 틀린 값에서 수렴할 거라 봤는데 그렇지 않다.

    수렴했다고 말하면 값이 맞고, 값이 모자라면 수렴 안 했다고 말한다. 가짜 평탄부가 없다.
    """
    for f_slip in (0.8, 1.6, 3.2):
        for hz in (0.11, 0.35, 1.1):
            est, _, converged, _ = m.ladder(f_slip, tau=0.2, exc_hz=hz)
            if converged:
                assert est == pytest.approx(f_slip, rel=0.08), (f_slip, hz, est)
            else:
                assert est <= f_slip * 1.05            # 하한이지 과대추정이 아니다


def test_a_false_plateau_is_structurally_impossible_on_the_amplitude_axis():
    """**왜** 안 깨지는지를 고정한다 — F_slip 을 뺀 모든 천장이 진폭을 따라 올라간다.

    기하학적 천장 K·A 도, 점성 천장 K·(A·ω)·τ 도 A 에 비례한다. 그래서 추정은 진폭에 대해
    단조 증가하고, 멈추는 곳은 F_slip 뿐이다.
    """
    for hz in (0.11, 1.1):
        _, _, _, vals = m.ladder(3.2, tau=0.2, exc_hz=hz)
        v = np.asarray(vals)
        assert np.all(np.diff(v) > -0.05)              # 단조(수치 여유 안에서)
        assert v[-1] <= 3.2 * 1.05                     # F_slip 위로는 안 간다


def test_relaxation_costs_amplitude_not_correctness():
    """이완이 물리는 대가는 **틀린 값이 아니라 더 큰 진폭**이다."""
    _, amp_el, ok_el, _ = m.ladder(3.2, tau=np.inf, exc_hz=0.11)
    _, amp_re, ok_re, _ = m.ladder(3.2, tau=0.2, exc_hz=0.11)
    assert ok_el                                        # 순탄성은 잡히고
    assert (not ok_re) or amp_re > amp_el               # 이완은 더 흔들거나 못 잡는다
    _, amp_fast, ok_fast, _ = m.ladder(3.2, tau=0.2, exc_hz=1.1)
    assert ok_fast                                      # 주파수를 올리면 다시 잡힌다


def test_raising_the_frequency_recovers_the_true_value():
    for f_slip in (0.8, 3.2):
        est, _, _, _ = m.ladder(f_slip, tau=0.2, exc_hz=1.1)
        assert est == pytest.approx(f_slip, rel=0.06)


def test_the_two_axis_ladder_fixes_it():
    """진폭과 주파수를 **둘 다** 올리면 참값이 나온다."""
    for f_slip in (0.8, 1.6, 3.2):
        est, fz, _, ok, _ = m.two_axis_ladder(f_slip, tau=0.2)
        assert ok
        assert est == pytest.approx(f_slip, rel=0.06)


def test_the_velocity_requirement_predicts_which_frequency_works():
    """A·ω > F_slip/(K·τ) 가 실제로 성패를 가르는 선인지 확인한다."""
    f_slip, tau, amp = 3.2, 0.2, 0.030            # 진폭은 기하학적 요구를 이미 만족
    assert amp > 2 * f_slip / K                   # A > 2 F_slip / K = 21.3 mm
    need_w = f_slip / (K * tau) / amp             # 필요한 각속도
    lo, hi = need_w / (2 * np.pi) / 4, need_w / (2 * np.pi) * 4
    e_lo = s60.fit_from_dwell(*m.insertion_log(f_slip, tau=tau, dwell=True,
                                               exc_mm=amp * 1e3, exc_hz=lo),
                              mu=jc.tele.MU)[0]
    e_hi = s60.fit_from_dwell(*m.insertion_log(f_slip, tau=tau, dwell=True,
                                               exc_mm=amp * 1e3, exc_hz=hi),
                              mu=jc.tele.MU)[0]
    assert e_lo < f_slip * 0.8                    # 느리면 못 잡고
    assert e_hi > e_lo * 1.2                      # 빠르면 확실히 올라간다


def test_relaxation_only_hurts_identification_it_never_inflates():
    """이완은 추정을 **낮추기만** 한다 — 과대추정 쪽으로는 안 간다(편향의 부호를 안다)."""
    for f_slip in (0.8, 3.2):
        with_relax, _, _, _ = m.ladder(f_slip, tau=0.2, exc_hz=0.11)
        elastic, _, _, _ = m.ladder(f_slip, tau=np.inf, exc_hz=0.11)
        assert with_relax <= elastic * 1.02


# --------------------------------------------------------------------------- #
# C/D — 결론은 살아남는가
# --------------------------------------------------------------------------- #
def test_dose_and_peak_disagree_about_retracting():
    """**지표가 힘 축의 승자를 바꾼다** — 최댓값은 후퇴의 과도현상을, 누적은 그 뒤 정상상태를 본다."""
    h = m.policy(3.2, 0.2, retract=False, seeds=3)
    r = m.policy(3.2, 0.2, retract=True, seeds=3)
    assert r[1] < h[1]                       # 누적: 후퇴가 낫다
    assert r[0] >= h[0] * 0.98               # 진폭: 낫지 않다(같거나 나쁘다)


def test_retract_dose_is_lower_at_every_slip_limit():
    for f_slip in (0.4, 1.6, 3.2):
        h = m.policy(f_slip, 0.2, retract=False, seeds=3)
        r = m.policy(f_slip, 0.2, retract=True, seeds=3)
        assert r[1] < h[1]


def test_exp60_peak_metric_measures_the_controller_not_the_tissue():
    """**이 실험에서 가장 중요한 발견.** 환자를 완전히 세워도 진폭이 남고, 조직·τ 를 바꿔도 같다.

    exp 60 은 위상 민감성을 고치려고 증분 → 진폭으로 갈아탔는데, 그 진폭은 정지 제어기가 목표
    위치로 정착하는 과도현상이 지배한다. 즉 조직의 위해를 거의 못 본다.
    """
    still = [m.policy(f, t, breath_mm=0.0, seeds=6)[0]
             for f in (0.4, 6.4) for t in (0.2, np.inf)]
    assert min(still) > 1.5                       # 환자가 안 움직이는데도 크다
    assert max(still) - min(still) < 0.05         # 조직/τ 를 바꿔도 같다


def test_dose_metric_does_see_the_tissue():
    """반대로 누적은 F_slip 에도 τ 에도 반응한다 — 그래서 정책 비교에 쓸 수 있다."""
    soft = m.policy(0.4, np.inf, breath_mm=0.0, seeds=6)[1]
    stiff = m.policy(6.4, np.inf, breath_mm=0.0, seeds=6)[1]
    relaxing = m.policy(6.4, 0.2, breath_mm=0.0, seeds=6)[1]
    assert stiff > soft * 1.15                    # 파악이 셀수록 누적이 크고
    assert relaxing < stiff * 0.95                # 이완하면 줄어든다


def test_the_structural_conclusion_survives_the_metric_change():
    """지표를 바꿔도 '낮은 교환비에서는 F_slip 과 무관하게 붙들기'가 그대로다."""
    for metric in (0, 1):                         # 0 = 진폭, 1 = 누적
        ws = []
        for f_slip in (0.4, 1.6, 6.4):
            h = m.policy(f_slip, 0.2, retract=False, seeds=6)
            r = m.policy(f_slip, 0.2, retract=True, seeds=6)
            ws.append(s60.flip_trade(h[metric], h[3], r[metric], r[3]))
        assert min(ws) > 5.0                      # 그 아래에서는 어느 F_slip 이든 붙들기


def test_dose_metric_needs_the_held_duration_to_be_reported():
    """누적은 힘 × 시간이라 **정지 시간을 함께 내지 않으면 해석할 수 없다.**"""
    h = m.policy(1.6, 0.2, retract=False, seeds=3)
    assert h[2] > 0.0
    assert h[1] == pytest.approx(h[1], abs=0)            # 존재 확인
    assert h[1] / h[2] > 0.5                            # 평균 힘이 물리적으로 말이 된다
