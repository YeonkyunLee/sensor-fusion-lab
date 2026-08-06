"""exp 63 의 주장과 **틀린 예측**을 회귀로 고정한다.

들어가며 "끌림 축에서는 후퇴가 이긴다"고 예측했고 틀렸다. 그게 기록으로 남아야 하는 종류라
테스트로 박아 둔다. 실제 결과는 정책이 아니라 **정보의 값**이 뒤집힌 것이다 —
힘 축에서 평평하던 F_slip 이 변형 축에서는 결정을 가른다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("63_harm_is_not_force")
s59 = import_module("59_what_is_safe_state")
s61 = import_module("61_tissue_relaxes")
jc = import_module("56_jittery_channel")


# --------------------------------------------------------------------------- #
# 앞 실험을 무효화하지 않는가
# --------------------------------------------------------------------------- #
def test_drag_counting_does_not_change_the_force():
    """끌림 적산은 **관측일 뿐** 물리를 바꾸면 안 된다."""
    a = m.DraggingTissue(f_slip=0.8, tau=m.TAU)
    b = s61.RelaxingTissue(f_slip=0.8, tau=m.TAU)
    for x in np.linspace(jc.X_SURFACE - 0.002, jc.X_SURFACE + 0.06, 800):
        assert a.force(x) == pytest.approx(b.force(x), abs=1e-12)
    assert a.drag > 0.0


def test_pre_puncture_motion_is_not_counted_as_damage():
    """관통 전에는 앵커를 도구에 붙여 두므로 접근 구간이 손상으로 잡히면 안 된다."""
    ts = m.DraggingTissue(f_slip=0.8, tau=m.TAU)
    for x in np.linspace(jc.X_SURFACE, jc.X_SURFACE + 0.004, 300):
        ts.force(x)
        if ts.punctured:
            break
    assert ts.drag == pytest.approx(0.0, abs=1e-12)


def test_runs_without_a_drag_counting_tissue_are_unaffected():
    """`drag` 를 세지 않는 조직에서는 지표가 0 이고 나머지는 그대로여야 한다."""
    kw = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS, estop=True,
              resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=m.BREATH_HZ)
    r = m.bc.run("tdpa", seed=0, tissue_obj=s59.GrippingTissue(), **kw)
    assert r["drag_held_mm"] == 0.0 and r["drag_total_mm"] == 0.0
    assert r["f_e_held_swing"] > 0.0


# --------------------------------------------------------------------------- #
# 왜 힘이 이걸 못 보는가 (구조적 이유)
# --------------------------------------------------------------------------- #
def test_force_is_pinned_while_the_tissue_keeps_being_dragged():
    """**핵심 메커니즘.** 정상 미끄러짐 중 힘은 F_slip 에 고정인데 앵커는 계속 간다."""
    ts = m.DraggingTissue(f_slip=0.8, tau=np.inf)      # 이완 없이도 성립한다
    x = jc.X_SURFACE
    for _ in range(4000):                               # 관통 + 정상 미끄러짐까지
        x += 0.04 / 4000
        ts.force(x)
    f0, d0 = -ts.force(x), ts.drag
    for _ in range(2000):                               # 계속 등속 전진
        x += 1e-5
        ts.force(x)
    f1, d1 = -ts.force(x), ts.drag
    assert (d1 - d0) > 0.015                            # 조직은 15 mm 넘게 끌렸는데
    assert abs(f1 - f0) < 0.4                           # 힘은 거의 그대로다


def test_relaxation_drags_the_tissue_with_no_tool_motion_at_all():
    """도구가 완전히 멈춰 있어도 이완이 앵커를 끈다 — 힘은 오히려 내려간다."""
    ts = m.DraggingTissue(f_slip=0.8, tau=0.2)
    x = jc.X_SURFACE
    for _ in range(4000):
        x += 0.04 / 4000
        ts.force(x)
    f0, d0 = -ts.force(x), ts.drag
    for _ in range(2000):
        ts.force(x)                                     # 도구 정지
    assert -ts.force(x) < f0                            # 힘은 줄고
    assert ts.drag > d0                                 # 변형은 늘었다


def test_holding_still_accumulates_drag_while_force_settles():
    """조직만 떼어낸 대조 — 힘은 안정되는데 끌림은 계속 쌓인다."""
    f, d = m.hold_trace(secs=4.0)
    assert d[-1] > 5.0                                  # mm 단위로 크게 쌓이고
    assert np.max(f[len(f) // 2:]) <= np.max(f) + 1e-9  # 힘은 더 커지지 않는다
    half = d[len(d) // 2]
    assert d[-1] > half * 1.3                           # 후반에도 계속 쌓인다


# --------------------------------------------------------------------------- #
# 틀린 예측 — 기록으로 고정
# --------------------------------------------------------------------------- #
def test_my_prediction_failed_retracting_does_not_win_on_drag():
    """**들어가며 한 예측이 틀렸다.** 후퇴 자체가 조직을 끌기 때문이다.

    기본 파악(0.8 N)에서는 끌림 축에서도 붙들기가 이긴다.
    """
    h = m.series(seeds=6)
    r = m.series(retract=True, seeds=6)
    d = r["drag_held_mm"] - h["drag_held_mm"]
    wins = int(np.nansum(d < 0))
    assert wins <= 6 // 2                               # 후퇴가 과반을 못 넘는다
    assert np.nanmedian(d) > 0.0                        # 오히려 더 끈다


def test_the_four_metrics_do_not_agree_with_each_other():
    """네 지표가 갈린다 — 축을 하나 더 열어도 해소되지 않는다."""
    h = m.series(seeds=6)
    r = m.series(retract=True, seeds=6)
    winners = set()
    for key, _, _ in m.METRICS:
        d = r[key] - h[key]
        winners.add("retract" if int(np.nansum(d < 0)) > 3 else "hold")
    assert len(winners) == 2, "지표가 전부 같은 답을 주면 이 실험의 전제가 무너진다"


# --------------------------------------------------------------------------- #
# 실제 결과 — 정보의 값이 축에 걸려 있었다
# --------------------------------------------------------------------------- #
def test_force_swing_is_flat_in_the_slip_limit_but_drag_is_not():
    """힘 진폭은 F_slip 전 구간에서 평평하고(exp 60·61 의 포화), 끌림은 감소한다.

    시드 수에 주의: 12 시드 중앙값에서 진폭은 세 값이 **완전히 같다**. 적은 시드로는 흔들린다.
    """
    sw, dg = [], []
    for f_slip in (0.4, 1.6, 6.4):
        h = m.series(f_slip=f_slip, seeds=12)
        sw.append(float(np.nanmedian(h["f_e_held_swing"])))
        dg.append(float(np.nanmedian(h["drag_held_mm"])))
    assert max(sw) - min(sw) < 0.05 * max(sw)           # 힘: 평평
    assert dg[0] > dg[-1] * 1.2                         # 끌림: 확실히 감소
    # 끌림도 결국 **포화**한다(파악이 충분히 세면 아예 안 미끄러지므로). 단조가 아니라
    # '감소 후 포화'가 맞는 서술이다 — 1.6 N 과 6.4 N 사이는 0.2% 안에서 같다.
    assert dg[0] > dg[1] * 1.15                         # 앞 구간에서 확실히 떨어지고
    assert abs(dg[1] - dg[2]) < 0.03 * dg[1]            # 뒤 구간에서 포화한다


def test_a_stronger_grip_is_protective_on_the_drag_axis():
    """**부호가 반대다** — 힘으로 보면 무관/나쁜 것이 변형으로 보면 보호적이다."""
    weak = m.series(f_slip=0.4, seeds=6)
    strong = m.series(f_slip=6.4, seeds=6)
    assert np.nanmedian(strong["drag_held_mm"]) < np.nanmedian(weak["drag_held_mm"])
    assert (np.nanmedian(strong["f_e_held_swing"])
            >= np.nanmedian(weak["f_e_held_swing"]) - 1e-9)


def test_on_the_drag_axis_the_policy_winner_flips_with_the_slip_limit():
    """**이 실험의 결과.** exp 60 의 '재도 결정이 안 바뀐다'는 힘 축 위에서만 참이었다."""
    def winner(f_slip):
        h = m.series(f_slip=f_slip, seeds=6)
        r = m.series(f_slip=f_slip, retract=True, seeds=6)
        return ("retract" if np.nanmedian(r["drag_held_mm"])
                < np.nanmedian(h["drag_held_mm"]) else "hold")
    assert winner(0.4) == "hold"
    assert winner(6.4) == "retract"


def test_drag_is_a_different_axis_not_a_third_force_metric():
    """힘 지표끼리는 서로 닮았고 끌림과는 안 닮았다 — 사각지대가 실재한다."""
    pool = {k: np.concatenate([m.series(f_slip=f, seeds=12)[k] for f in (0.4, 1.6, 6.4)])
            for k, _, _ in m.METRICS}

    def rho(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        ra = np.argsort(np.argsort(a[ok])).astype(float)
        rb = np.argsort(np.argsort(b[ok])).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))

    force = [k for k, _, _ in m.METRICS[:3]]
    ff = [rho(pool[a], pool[b]) for i, a in enumerate(force) for b in force[i + 1:]]
    fd = [rho(pool[a], pool["drag_held_mm"]) for a in force]
    assert min(ff) > 0.4, f"힘 지표끼리는 서로 닮아야 한다: {ff}"
    assert max(fd) < min(ff), f"힘끼리 {ff} vs 힘-끌림 {fd}"
    assert np.median(fd) < 0.3, f"끌림은 힘 지표와 다른 축이어야 한다: {fd}"


def test_drag_is_not_just_a_proxy_for_held_time():
    """끌림이 정지 시간의 재포장이면 새 정보가 아니다 — 상관이 낮아야 한다."""
    h = m.series(seeds=12)
    ok = np.isfinite(h["drag_held_mm"]) & np.isfinite(h["secs_held"])
    r = float(np.corrcoef(h["drag_held_mm"][ok], h["secs_held"][ok])[0, 1])
    assert abs(r) < 0.6
