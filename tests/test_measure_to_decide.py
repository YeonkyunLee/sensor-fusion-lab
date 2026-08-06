"""exp 60 의 설계 함정을 회귀로 고정한다.

이 실험에서 실제로 밟은 것들: 여기 주파수를 호흡과 같게 잡아 상대 운동이 상쇄된 일, 관통 전
비선형 구간까지 회귀에 넣어 상수항이 무의미해진 일, 한 번의 고리 모양으로 포화를 판정하려다
되붙음 리플에 속은 일, 그리고 위해 지표를 정지 시점 대비 증분으로 잡아 정지 위상에 휘둘린 일.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("60_measure_to_decide")
s59 = import_module("59_what_is_safe_state")
jc = import_module("56_jittery_channel")


# --------------------------------------------------------------------------- #
# 조직 모델 — 앞 실험을 무효화하지 않는지
# --------------------------------------------------------------------------- #
def test_tunable_tissue_with_unit_scale_matches_exp59_exactly():
    """cut_scale=1.0 이면 exp 59 의 GrippingTissue 와 완전히 같아야 한다."""
    a = m.TunableTissue(k_grip=300.0, f_slip=0.8, cut_scale=1.0)
    b = s59.GrippingTissue(k_grip=300.0, f_slip=0.8)
    for x in np.linspace(jc.X_SURFACE - 0.002, jc.X_SURFACE + 0.05, 400):
        assert a.force(x) == pytest.approx(b.force(x), abs=1e-12)


def test_cut_scale_moves_only_the_cutting_term():
    """배율은 절삭 기저에만 걸리고 파악 항은 건드리지 않아야 한다."""
    one = m.TunableTissue(f_slip=0.8, cut_scale=1.0)
    two = m.TunableTissue(f_slip=0.8, cut_scale=2.0)
    xs = np.linspace(jc.X_SURFACE, jc.X_SURFACE + 0.04, 300)
    f1 = np.array([one.force(x) for x in xs])
    f2 = np.array([two.force(x) for x in xs])
    deep = xs > jc.X_SURFACE + 0.02
    # 관통 후 깊은 구간에서는 둘 다 미끄러지는 중이라 파악 항이 같은 상수다.
    # 따라서 차이는 절삭 기저 하나뿐이고, 부호가 일정해야 한다.
    diff = f2[deep] - f1[deep]
    assert np.all(diff < 0)                       # 더 강한 저항(음수 방향)
    assert diff.std() / abs(diff.mean()) < 0.6    # 파악 항이 차이에 안 섞임


# --------------------------------------------------------------------------- #
# C — 삽입 로그의 교락
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("f_slip", [0.4, 0.8, 1.6, 3.2])
def test_insertion_constant_is_exactly_cutting_plus_slip(f_slip):
    """**핵심 주장**: 삽입만 보면 상수항이 F_cut + F_slip 이라 둘이 안 갈라진다."""
    d, f, _ = m.insertion_log(f_slip, dwell=False)
    a, _ = m.fit_from_insertion(d, f)
    assert a == pytest.approx(jc.tele.F_CUT + f_slip, rel=0.02)


@pytest.mark.parametrize("f_slip", [0.4, 3.2])
def test_insertion_slope_is_friction_and_independent_of_slip(f_slip):
    """반대로 기울기는 깨끗하다 — 그래서 삽입 로그를 버리지 않고 보정에 쓴다."""
    d, f, _ = m.insertion_log(f_slip, dwell=False)
    _, mu = m.fit_from_insertion(d, f)
    assert mu == pytest.approx(jc.tele.MU, rel=0.05)


def test_fit_must_exclude_the_pre_puncture_region():
    """관통 전 비선형 강성을 넣으면 상수항이 무너진다(처음에 그렇게 짜서 값이 다 틀렸다)."""
    d, f, _ = m.insertion_log(3.2, dwell=False)
    good, _ = m.fit_from_insertion(d, f)
    A = np.stack([np.ones_like(d), d], axis=1)
    naive = float(np.linalg.lstsq(A, f, rcond=None)[0][0])
    truth = jc.tele.F_CUT + 3.2
    assert abs(good - truth) < abs(naive - truth) / 3.0


# --------------------------------------------------------------------------- #
# D — 여기 설계
# --------------------------------------------------------------------------- #
def test_excitation_at_the_breathing_frequency_cancels_itself():
    """같은 주파수로 흔들면 상대 운동이 사라진다 — 여기를 키웠는데 신호가 0 이 되는 함정."""
    ts = m.TunableTissue(f_slip=0.8)
    for x in np.linspace(jc.X_SURFACE, jc.X_SURFACE + 0.06, 4000):
        ts.force(x)
    t = np.arange(6000) * jc.DT
    same = 5e-3 * np.sin(2 * np.pi * s59.BREATH_HZ * t) - \
        5e-3 * np.sin(2 * np.pi * s59.BREATH_HZ * t)
    diff = 5e-3 * np.sin(2 * np.pi * 0.11 * t) - \
        5e-3 * np.sin(2 * np.pi * s59.BREATH_HZ * t)
    assert np.ptp(same) == pytest.approx(0.0, abs=1e-12)
    assert np.ptp(diff) > 5e-3


def test_breathing_only_estimate_is_capped_by_stiffness_times_amplitude():
    """호흡만 쓰면 참값과 무관하게 K_grip x A / 2 에서 잘린다 — 조직이 아니라 호흡을 잰 것."""
    cap = m.K_GRIP0 * (2 * 5.0e-3) / 2.0
    for f_slip in (3.2, 6.4):
        d, f, mk = m.insertion_log(f_slip, dwell=True, exc_mm=0.0)
        est, _ = m.fit_from_dwell(d, f, mk, mu=jc.tele.MU)
        assert est < f_slip * 0.6                 # 참값 근처에도 못 간다
        assert est == pytest.approx(cap, rel=0.15)


@pytest.mark.parametrize("f_slip", [0.4, 0.8, 1.6, 3.2, 6.4])
def test_the_amplitude_ladder_recovers_the_true_slip_limit(f_slip):
    """사다리를 끝까지 올리면 참값이 나온다(하한에서 멈추지 않는다)."""
    est, amp, conv, _ = m.identify_ladder(f_slip)
    assert conv
    assert est == pytest.approx(f_slip, rel=0.05)


@pytest.mark.parametrize("f_slip", [0.8, 3.2])
def test_converged_amplitude_tracks_two_fslip_over_kgrip(f_slip):
    """수렴 진폭이 예측식 2·F_slip/K_grip 을 사다리 한 칸 안에서 따라가야 한다."""
    need = 2.0 * f_slip / m.K_GRIP0 * 1e3
    _, amp, conv, _ = m.identify_ladder(f_slip)
    assert conv
    idx = m.LADDER.index(amp)
    lower = m.LADDER[max(idx - 1, 0)]
    assert lower <= need <= m.LADDER[min(idx + 1, len(m.LADDER) - 1)] * 1.5


def test_ladder_is_monotone_so_convergence_means_something():
    """추정은 진폭에 대해 단조 증가해야 한다 — 그래야 '안 자라면 참값'이 성립한다."""
    _, _, _, vals = m.identify_ladder(3.2)
    v = np.asarray(vals)
    assert np.all(np.diff(v) > -0.05)


# --------------------------------------------------------------------------- #
# D2 — 잡음 편향의 부호
# --------------------------------------------------------------------------- #
def test_sensor_noise_biases_the_estimate_upward_not_randomly():
    """최댓값-최솟값 차는 잡음을 **한쪽으로만** 흡수한다. 부호를 알아야 보정할 수 있다."""
    errs = []
    for noise in (0.0, 0.05):
        e = []
        for s in range(4):
            d, f, mk = m.insertion_log(0.4, dwell=True, exc_mm=20.0,
                                       noise_N=noise, seed=s)
            est, _ = m.fit_from_dwell(d, f, mk, mu=jc.tele.MU)
            e.append(est - 0.4)
        errs.append(float(np.median(e)))
    assert errs[0] == pytest.approx(0.0, abs=0.05)
    assert errs[1] > 0.05                          # 과대추정 쪽으로만 간다


# --------------------------------------------------------------------------- #
# A/B — 위해 지표와 정보의 값
# --------------------------------------------------------------------------- #
def test_hold_load_metric_must_not_depend_on_the_stop_phase():
    """정지 시점 대비 증분은 위상에 휘둘려 **환자가 더 움직일수록 부하가 작아지는** 착시를 만든다.

    진폭 지표는 환자 움직임이 커지면 커져야 한다 — 그게 물리다.
    """
    swings, incs = [], []
    for breath in (2.0, 10.0):
        sw, inc = [], []
        for s in range(3):
            r = m.bc.run("tdpa", seed=s, tail_ms=s59.TAIL_MS, loss=0.10,
                         burst_len=s59.BURST_MS, estop=True, resume_ms=60.0,
                         blind_mm=1.0, breath_mm=breath, breath_hz=s59.BREATH_HZ,
                         tissue_obj=m.TunableTissue(f_slip=3.2))
            sw.append(r["f_e_held_swing"]); inc.append(r["df_held_max"])
        swings.append(float(np.median(sw))); incs.append(float(np.median(inc)))
    assert swings[1] > swings[0]                   # 진폭 지표는 물리를 따른다
    assert incs[1] < incs[0]                       # 증분 지표는 뒤집힌다(그래서 안 쓴다)


def test_the_grip_saturates_so_a_larger_slip_limit_changes_nothing():
    """K_grip x 상대운동 위로는 F_slip 이 시스템에 아무 영향도 주지 않는다.

    이게 '못 재는 이유와 안 중요한 이유가 같다'의 실체다.
    """
    a = m.policy(3.2, 5.0, seeds=3)
    b = m.policy(6.4, 5.0, seeds=3)
    assert a[0] == pytest.approx(b[0], rel=0.02)
    assert a[1] == pytest.approx(b[1], rel=0.02)


def test_retracting_gets_worse_as_the_grip_gets_stronger():
    """**exp 59 의 뒤집힘 조건을 정정한다.** 후퇴도 같은 파악 항을 거슬러 끈다."""
    soft = m.policy(0.4, 5.0, retract=True, seeds=3)[0]
    stiff = m.policy(3.2, 5.0, retract=True, seeds=3)[0]
    assert stiff > soft


def test_no_measurement_of_the_tissue_flips_the_decision_at_low_exchange_rates():
    """정보의 값 — 맹행이 비싼 쪽에서는 어떤 F_slip 이어도 붙들기가 이긴다."""
    for f_slip in (0.4, 3.2):
        lh, bh = m.policy(f_slip, 5.0, retract=False, seeds=3)
        lr, br = m.policy(f_slip, 5.0, retract=True, seeds=3)
        w_star = m.flip_trade(lh, bh, lr, br)
        assert w_star > 5.0


def test_flip_trade_reports_infinity_when_retracting_also_costs_more_force():
    """힘까지 더 나쁘면 어떤 교환비에서도 안 이긴다 — 경계가 '있다'고 말하면 안 된다."""
    assert m.flip_trade(1.0, 2.0, 1.5, 8.0) == np.inf
    assert np.isfinite(m.flip_trade(2.0, 2.0, 1.0, 8.0))
