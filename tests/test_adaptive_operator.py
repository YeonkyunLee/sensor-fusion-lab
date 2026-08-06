"""exp 62 의 주장과 함정을 회귀로 고정한다.

가장 중요한 것은 **완주 지표 없이 술자 계층을 비교하면 전부 좋아 보인다**는 것이다. 그건 exp 56 이
만든 R18 이 자기 저장소에 다시 걸린 사례라, 테스트로 못박아 둔다. 그리고 각 계층이 앞 계층에
**하나만** 더한 것인지, 아무것도 안 켜면 앞 실험 경로가 그대로인지도 고정한다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("62_adaptive_operator")
s59 = import_module("59_what_is_safe_state")
jc = import_module("56_jittery_channel")


# --------------------------------------------------------------------------- #
# 앞 실험을 무효화하지 않는가
# --------------------------------------------------------------------------- #
def test_all_new_operator_switches_default_off():
    """새 인자를 안 주면 exp 59 경로와 **비트 단위로** 같아야 한다."""
    kw = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS, estop=True,
              resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=s59.BREATH_HZ,
              op_react_ms=200.0)
    a = m.bc.run("tdpa", seed=1, tissue_obj=s59.GrippingTissue(), **kw)
    b = m.bc.run("tdpa", seed=1, tissue_obj=s59.GrippingTissue(),
                 op_force_N=0.0, op_learn=0.0, op_reverse_mm=0.0, **kw)
    for k in ("final_depth_mm", "resume_vmax_mms", "blind_max_mm", "e_min"):
        assert a[k] == pytest.approx(b[k], abs=1e-12), k


def test_operator_clock_runs_at_full_rate_without_learning():
    """학습을 안 켜면 술자의 내부 시계가 실제 시간과 같이 간다(얼어 있던 동안만 뺀다)."""
    r = m.bc.run("tdpa", seed=0, tail_ms=s59.TAIL_MS, loss=0.10,
                 burst_len=s59.BURST_MS, estop=True, resume_ms=60.0, blind_mm=1.0,
                 tissue_obj=s59.GrippingTissue())
    assert r["op_rate_end"] == pytest.approx(1.0)
    assert r["t_op_end"] == pytest.approx(jc.STEPS * jc.DT, rel=1e-6)


# --------------------------------------------------------------------------- #
# 각 계층이 실제로 작동하는가
# --------------------------------------------------------------------------- #
def test_force_perception_actually_fires():
    """힘 단서가 실제로 걸려야 '이득이 없다'는 결론이 의미가 있다(안 걸려서 없는 게 아니다)."""
    kw = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS, estop=True,
              resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=s59.BREATH_HZ,
              op_react_ms=m.REACT_MS)
    off = m.bc.run("tdpa", seed=0, tissue_obj=s59.GrippingTissue(), **kw)
    on = m.bc.run("tdpa", seed=0, tissue_obj=s59.GrippingTissue(),
                  op_force_N=m.FORCE_N, **kw)
    assert off["n_force_cue"] == 0
    assert on["n_force_cue"] > 100


def test_learning_slows_the_operator_and_recovers():
    """학습은 겪을 때마다 시계를 늦추고, 아무 일 없으면 천천히 회복해야 한다."""
    kw = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS, estop=True,
              resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=s59.BREATH_HZ,
              op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N)
    short = m.bc.run("tdpa", seed=0, tissue_obj=s59.GrippingTissue(),
                     op_learn=m.LEARN, steps=jc.STEPS, **kw)
    assert short["n_adverse"] > 0
    assert short["op_rate_end"] < 1.0                 # 늦춰졌고
    assert short["op_rate_end"] >= jc.OP_RATE_MIN     # 하한을 지킨다
    long = m.bc.run("tdpa", seed=0, tissue_obj=s59.GrippingTissue(),
                    op_learn=m.LEARN, steps=3 * jc.STEPS, **kw)
    assert long["op_rate_end"] > short["op_rate_end"]  # 시간이 있으면 회복한다


def test_reversal_pulls_the_target_back_not_just_freezes():
    """되돌림은 얼기와 달라야 한다 — **짝지어** 복귀 돌진이 낮아지는 것으로 확인한다."""
    froze = m.series(dict(op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N,
                          op_learn=m.LEARN), steps=3 * jc.STEPS)
    back = m.series(dict(op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N,
                         op_learn=m.LEARN, op_reverse_mm=m.REVERSE_MM),
                    steps=3 * jc.STEPS)
    dm, wins, tot = m.paired(froze, back)
    assert dm < 0.0
    assert wins >= tot * 2 // 3


# --------------------------------------------------------------------------- #
# 이 실험의 핵심 주장
# --------------------------------------------------------------------------- #
def test_the_tier_ladder_stops_completing_the_task():
    """**핵심.** 표준 길이에서는 계층이 올라갈수록 안전해 보이지만 **과제를 안 끝낸다.**

    exp 56 이 만든 R18 이 이 저장소 자신에게 다시 걸린 자리다.
    """
    t0 = m.tier({}, steps=jc.STEPS)
    t4 = m.tier(dict(op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N,
                     op_learn=m.LEARN, op_reverse_mm=m.REVERSE_MM), steps=jc.STEPS)
    assert t4["resume_vmax_mms"] < t0["resume_vmax_mms"] * 0.7    # 좋아 보이는데
    assert t4["final_depth_mm"] < t0["final_depth_mm"] * 0.85     # 과제를 안 끝낸다
    assert t0["final_depth_mm"] > 45.0                            # 기준선은 끝낸다


def test_learning_adds_nothing_once_the_task_has_to_finish():
    """완주를 맞추면 학습은 이득이 없다 — 힘 지각 위에 얹어도 나아지지 않는다."""
    long = 3 * jc.STEPS
    t2 = m.series(dict(op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N), steps=long)
    t3 = m.series(dict(op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N,
                       op_learn=m.LEARN), steps=long)
    assert np.nanmedian(t3["final_depth_mm"]) > 45.0
    dm, wins, tot = m.paired(t2, t3)
    assert wins <= tot * 2 // 3          # 대다수 시드에서 개선되지 않는다


def test_visual_reaction_is_real_but_not_universal():
    """exp 59 의 이득이 완주 조건에서도 남는다 — 단 **8/12 정도이지 보편적이지 않다.**"""
    long = 3 * jc.STEPS
    t0 = m.series({}, steps=long)
    t1 = m.series(dict(op_react_ms=m.REACT_MS), steps=long)
    assert np.nanmedian(t0["final_depth_mm"]) > 45.0
    assert np.nanmedian(t1["final_depth_mm"]) > 45.0
    dm, wins, tot = m.paired(t0, t1)
    assert dm < -10.0                    # 짝지은 이득이 실재하고
    assert wins < tot                    # 그런데 모든 시드는 아니다


def test_force_perception_buys_reliability_not_magnitude():
    """**이 실험의 핵심 발견.** 중복 단서는 천장을 올리지 않고 **실패를 없앤다.**

    6 시드 중앙값으로는 "이득 없음"으로 보였다 — exp 59 가 잡았던 함정에 다시 빠진 자리라
    짝지은 통계로 고정해 둔다.
    """
    long = 3 * jc.STEPS
    t0 = m.series({}, steps=long)
    t1 = m.series(dict(op_react_ms=m.REACT_MS), steps=long)
    t2 = m.series(dict(op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N), steps=long)
    _, w1, tot = m.paired(t0, t1)
    _, w2, _ = m.paired(t0, t2)
    assert w2 > w1                        # 개선되는 시드가 늘어난다
    # 그리고 시각 단서가 이미 이기던 시드에서는 값이 **바뀌지 않는다**(힘 단서가 안 걸린 것).
    same = np.isclose(t1["resume_vmax_mms"], t2["resume_vmax_mms"], rtol=1e-9)
    assert int(np.sum(same)) >= 3


def test_reversal_costs_interruptions_not_blind_travel():
    """되돌림의 대가는 **맹행이 아니다** — 짝지어 보면 개입 횟수와 붙들기 힘이다.

    6 시드 중앙값으로는 맹행이 대가로 보였는데, 그것도 아티팩트였다.
    """
    long = 3 * jc.STEPS
    t1 = m.series(dict(op_react_ms=m.REACT_MS), steps=long)
    t4 = m.series(dict(op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N,
                       op_learn=m.LEARN, op_reverse_mm=m.REVERSE_MM), steps=long)
    d_res, w_res, tot = m.paired(t1, t4)
    assert d_res < 0 and w_res >= tot * 2 // 3           # 복귀는 확실히 좋아지고
    d_adv, w_adv, _ = m.paired(t1, t4, key="n_adverse", lower_is_better=False)
    assert d_adv > 0 and w_adv >= tot * 2 // 3           # 개입은 확실히 늘고
    d_bl, w_bl, _ = m.paired(t1, t4, key="blind_max_mm", lower_is_better=False)
    assert w_bl <= tot * 2 // 3                          # 맹행은 대가가 아니다


def test_master_lock_still_loses_in_every_operator_model():
    """**exp 59 의 결론을 다섯 술자 모델 전부에서 재확인한다.** 내 예측(힘 지각이 구제한다)은 틀렸다.

    짝지어 재고, **보편적이지 않다**(시드의 2/3 정도)는 것도 같이 고정한다.
    """
    long = 3 * jc.STEPS
    for _, kw in m.TIERS:
        free = m.series(kw, lock=False, steps=long)
        lock = m.series(kw, lock=True, steps=long)
        dm, worse, tot = m.paired(free, lock, lower_is_better=False)
        assert dm > 0.0, kw                              # 잠그면 복귀가 나빠지고
        assert worse >= tot // 2, kw                     # 과반 시드에서 그렇다
        assert (np.nanmedian(lock["mismatch_release_mm"])
                < np.nanmedian(free["mismatch_release_mm"])), kw   # 어긋남은 줄어드는데도


def test_the_felt_force_includes_the_master_lock():
    """잠금의 저항도 손에 느껴져야 '잠금이 단서를 가리는가'라는 질문이 공정하다."""
    kw = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS, estop=True,
              resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=s59.BREATH_HZ,
              op_react_ms=m.REACT_MS, op_force_N=m.FORCE_N, steps=3 * jc.STEPS)
    free = m.bc.run("tdpa", seed=0, master_lock=False,
                    tissue_obj=s59.GrippingTissue(), **kw)
    lock = m.bc.run("tdpa", seed=0, master_lock=True,
                    tissue_obj=s59.GrippingTissue(), **kw)
    assert lock["n_force_cue"] != free["n_force_cue"]


def test_a_reacting_operator_does_not_break_channel_passivity():
    """사람은 증명 밖에 있지만, **규칙 형태의 반응**은 에너지를 넣지 않는다."""
    long = 3 * jc.STEPS
    for _, kw in m.TIERS:
        r = m.tier(kw, steps=long)
        assert r["e_min"] >= -1e-9, kw
        assert r["diverged"] == 0, kw
