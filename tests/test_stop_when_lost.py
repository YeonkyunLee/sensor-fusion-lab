"""exp 58 — 통신 상실 정지: exp 57 이 '미해결'로 남긴 것을 설계해 닫은 것.

이 실험은 앞 실험의 **틀린 진단을 보완**하는 것이라, 테스트도 세 갈래를 지킨다.
  1) 절제가 실제로 "지금의 경계는 우연"임을 보이는가 (보완의 근거).
  2) 정지가 **설계된 경계**를 만드는가 — 어느 항이 살아 있는지와 무관하게, 그리고
     '안전하지만 못 쓰는' 쪽으로 퇴화하지 않으면서.
  3) exp 57 에서 실패한 처방(λ 게이팅)이 이제 해가 되지 않는가 = R20 의 검증.
  그리고 설계 중에 밟은 함정 두 개(순간 트리거·연속 fresh 요구)를 회귀로 못박는다.
"""

from importlib import import_module
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
s58 = import_module("58_stop_when_lost")


# --------------------------------------------------------------------------- #
# 1) 보완의 근거 — 지금의 경계는 우연이다
# --------------------------------------------------------------------------- #
def test_the_bound_without_a_stop_depends_on_which_term_survives():
    """A 절 — 같은 채널·같은 암전인데 맹행이 한 자릿수 이상 벌어지면 그건 설계가 아니다."""
    full = s58.sweep("zoh", seeds=3)
    weak = s58.sweep("zoh", seeds=3, b_scale=0.3)
    assert weak["blind_max_mm"] > 3 * full["blind_max_mm"], \
        f"full {full['blind_max_mm']:.2f} vs weak damping {weak['blind_max_mm']:.2f} mm"


def test_self_limiting_weakens_when_the_damping_is_cut():
    """exp 57 이 찾은 자기 제한이 **플랜트의 성질**임을 확인한다(그래서 일반화 금지)."""
    full = s58.sweep("zoh", seeds=3)
    weak = s58.sweep("zoh", seeds=3, b_scale=0.3)
    assert np.isfinite(full["hold_late_um"]) and np.isfinite(weak["hold_late_um"])
    assert weak["hold_late_um"] > 3 * full["hold_late_um"], \
        "감쇠를 깎으면 오래 붙들고 있어도 계속 기어간다"


# --------------------------------------------------------------------------- #
# 2) 정지가 설계된 경계를 만드는가
# --------------------------------------------------------------------------- #
def test_the_stop_collapses_the_spread_across_ablations():
    """B 절의 핵심 — 평균이 아니라 **흔들림**이 좁아져야 설계된 경계다."""
    cases = [dict(), dict(tissue_on=False), dict(lam_pos=0.0), dict(b_scale=0.3)]
    off = [s58.sweep("zoh", seeds=3, **kw)["blind_max_mm"] for kw in cases]
    on = [s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS,
                    **kw)["blind_max_mm"] for kw in cases]
    assert max(off) / min(off) > 5, f"절제가 충분히 벌어져야 비교가 성립한다: {off}"
    assert max(on) / min(on) < 3, f"정지를 켜면 좁아져야 한다: {on}"
    assert max(on) < min(off) * 1.5


def test_the_stop_bounds_blind_travel_near_the_declared_margin():
    """경계가 **선언한 여유**에서 온다. 초과분은 정지 거리이므로 여유의 몇 배 안이면 된다."""
    for bm in (1.0, 2.0):
        s = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS,
                      blind_mm=bm)
        assert s["blind_max_mm"] < 3.5 * bm, \
            f"여유 {bm} mm 인데 맹행 {s['blind_max_mm']:.2f} mm"


def test_the_stop_does_not_degrade_into_safe_but_useless():
    """exp 56 의 0 채움이 빠졌던 함정 — 안전한데 과제를 못 하면 대책이 아니다."""
    no = s58.sweep("tdpa", seeds=3)
    yes = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS)
    assert yes["final_depth_mm"] > 0.9 * no["final_depth_mm"], \
        f"정지 없음 {no['final_depth_mm']:.1f} vs 정지 {yes['final_depth_mm']:.1f} mm"


def test_the_declared_margin_buys_the_stop_rate():
    """D 절 — 교환비가 통신이 아니라 해부에서 온다: 여유를 늘리면 덜 멈춘다."""
    tight = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS,
                      blind_mm=0.5)
    loose = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS,
                      blind_mm=4.0)
    assert tight["held_frac"] > 2 * loose["held_frac"]
    assert tight["blind_max_mm"] < loose["blind_max_mm"]


def test_the_stop_is_local_and_stays_passive():
    """집행이 채널을 거치지 않는 소산 항이므로 수동성 장부를 깨지 않아야 한다."""
    s = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS)
    assert s["e_min"] >= -1e-12, f"E_min = {s['e_min']:.3e} J"


# --------------------------------------------------------------------------- #
# 3) exp 57 의 규칙(R20) 검증
# --------------------------------------------------------------------------- #
def test_gating_the_drift_term_still_hurts_without_a_replacement():
    """exp 57 의 결과가 이 조건에서도 재현되어야 비교의 기준선이 성립한다."""
    bud = s58.sweep("tdpa", seeds=3)
    gated = s58.sweep("tdpa", seeds=3, lam_gate=True)
    assert gated["blind_max_mm"] > bud["blind_max_mm"], \
        f"예산만 {bud['blind_max_mm']:.2f} vs λ 게이트 {gated['blind_max_mm']:.2f} mm"


def test_once_the_brake_is_replaced_gating_no_longer_hurts():
    """**대체 먼저, 억제 나중** — 이 저장소가 exp 57 에서 세운 규칙의 실제 검증."""
    stop = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS)
    both = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=s58.RESUME_MS,
                     lam_gate=True)
    assert both["blind_max_mm"] <= max(stop["blind_max_mm"] * 1.15,
                                       stop["blind_max_mm"] + 0.2), \
        f"정지 {stop['blind_max_mm']:.2f} vs 정지+게이트 {both['blind_max_mm']:.2f} mm"
    assert both["final_depth_mm"] > 0.9 * stop["final_depth_mm"]


# --------------------------------------------------------------------------- #
# 설계 중 밟은 함정 — 회귀로 못박는다
# --------------------------------------------------------------------------- #
def test_trigger_must_integrate_travel_not_fire_on_the_instant():
    """첫 설계는 β=0 순간을 트리거로 썼다가 정지가 98.5% 걸려 과제를 못 했다.

    누적 거리 기준이면 여유를 아주 크게 잡으면 정지가 거의 안 걸려야 한다 — 순간 트리거였다면
    여유와 무관하게 계속 걸린다.
    """
    huge = s58.sweep("tdpa", seeds=2, estop=True, resume_ms=s58.RESUME_MS,
                     blind_mm=50.0)
    assert huge["held_frac"] < 0.05, f"정지 구간 {huge['held_frac']*100:.1f}%"
    assert huge["n_estop"] < 1.0


def test_release_must_not_require_consecutive_fresh_samples():
    """두 번째 함정 — 연속 fresh 를 요구하면 지터 채널에서 영구히 래치된다.

    램프가 길어도 결국 풀려서 과제를 완주해야 한다(정보가 오는 스텝에서만 램프가 오른다).
    """
    s = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=200.0)
    assert s["held_frac"] < 0.9, f"정지 구간 {s['held_frac']*100:.1f}%"
    assert s["final_depth_mm"] > 30.0, f"도달 깊이 {s['final_depth_mm']:.1f} mm"


def test_resume_peak_is_measured_in_a_fixed_window():
    """세 번째 함정 — 램프 구간만 보면 즉시 복귀가 가장 얌전해 보이는 착시가 생긴다.

    고정 창에서 재면 즉시 복귀가 **가장 빠르게** 튀어나가야 한다.
    """
    fast = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=0.0)
    slow = s58.sweep("tdpa", seeds=3, estop=True, resume_ms=200.0)
    assert fast["resume_vmax_mms"] > slow["resume_vmax_mms"], \
        f"즉시 {fast['resume_vmax_mms']:.0f} vs 램프 {slow['resume_vmax_mms']:.0f} mm/s"
    assert slow["held_frac"] > fast["held_frac"], "램프가 길면 멈춰 있는 시간이 늘어난다"


def test_run_is_reproducible():
    a = s58.run("tdpa", seed=1, estop=True, resume_ms=s58.RESUME_MS)
    b = s58.run("tdpa", seed=1, estop=True, resume_ms=s58.RESUME_MS)
    assert a["blind_max_mm"] == b["blind_max_mm"] and a["n_estop"] == b["n_estop"]


def test_earlier_experiments_are_untouched_by_the_new_switches():
    """56·57 의 기본 경로가 바뀌지 않았는지(정지·절제 손잡이는 모두 기본 off)."""
    jc = import_module("56_jittery_channel")
    a = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.10)
    b = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.10, estop=False,
               tissue_on=True, b_scale=1.0)
    assert a["e_min"] == b["e_min"] and a["final_depth_mm"] == b["final_depth_mm"]
    assert a["n_estop"] == 0 and a["held_frac"] == 0.0


def test_main_runs():
    out = s58.main(quick=True)
    assert set(out) == {"A", "B", "C", "D", "E"}
    assert Path("assets/58_stop_when_lost.png").exists()
