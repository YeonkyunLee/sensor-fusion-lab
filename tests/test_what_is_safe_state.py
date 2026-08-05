"""exp 59 — 정지가 안전 상태인가: 두 번 막히고, 막힌 이유가 결과였던 실험.

이 실험의 주장은 대부분 "제어가 아니라 **모델**이 문제였다"는 형태라, 테스트도 그 구조를 따른다.
  1) 옛 모델이 정말 그 위해를 표현하지 못하는가 (A 절 주장의 근거).
  2) 새 항이 그것을 표현하는가, 그리고 **움직일 때는 옛 모델로 환원되는가**(안 그러면 앞 실험들을
     무효화하는 변경이다).
  3) 정책 수준의 정직한 네거티브(후퇴가 이기지 않음)와 그 **뒤집힐 조건**.
  4) 술자 쪽에서 대책의 부호가 뒤집히는 것, 그리고 그 메커니즘(잠금이 신호를 가린다).
"""

from importlib import import_module
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
s59 = import_module("59_what_is_safe_state")
jc = import_module("56_jittery_channel")


# --------------------------------------------------------------------------- #
# 1) 옛 모델은 그 위해를 표현하지 못한다
# --------------------------------------------------------------------------- #
def test_the_old_tissue_model_cannot_express_holding_while_the_patient_moves():
    """A 절 — 도구를 고정하고 표면만 흔드는 최소 시험. 채널도 제어도 없다."""
    swing_old, _ = s59.hold_and_breathe(False, breath_mm=5.0)
    swing_new, _ = s59.hold_and_breathe(True, breath_mm=5.0)
    assert swing_old < 0.2, f"옛 모델 힘 변동 {swing_old:.3f} N (거의 0 이어야 한다)"
    assert swing_new > 5 * swing_old, \
        f"파악 항이 있어야 표현된다: {swing_old:.3f} → {swing_new:.3f} N"


def test_no_patient_motion_means_no_swing_in_either_model():
    """대조 — 환자가 안 움직이면 둘 다 변동이 없어야 한다(관측 대상이 그 움직임임을 확인)."""
    for grip in (False, True):
        swing, _ = s59.hold_and_breathe(grip, breath_mm=0.0)
        assert swing < 1e-6, f"grip={grip}: {swing:.3e} N"


def test_the_grip_contribution_saturates_at_the_slip_limit():
    """조직이 놓아주므로 위해가 유계다 — 안심의 근거인지 모델 한계인지를 구분해 적을 근거."""
    d5 = (s59.hold_and_breathe(True, breath_mm=5.0)[0]
          - s59.hold_and_breathe(False, breath_mm=5.0)[0])
    d10 = (s59.hold_and_breathe(True, breath_mm=10.0)[0]
           - s59.hold_and_breathe(False, breath_mm=10.0)[0])
    assert d10 < 2.2 * s59.F_SLIP, f"파악 몫 {d10:.3f} N 이 2×한계를 크게 넘었다"
    assert abs(d10 - d5) < 0.5, f"5→10 mm 에서 포화해야 한다: {d5:.3f} → {d10:.3f} N"


# --------------------------------------------------------------------------- #
# 2) 새 항이 앞 실험을 무효화하지 않는다
# --------------------------------------------------------------------------- #
def test_the_grip_term_reduces_to_the_old_model_while_advancing():
    """움직이는 동안은 계속 미끄러지므로 삽입 힘이 옛 모델과 사실상 같아야 한다.

    이게 깨지면 exp 47~58 의 숫자를 바꾸는 변경이 된다.
    """
    old, new = jc.tele.Tissue(), s59.GrippingTissue()
    x = s59.X_SURFACE
    fo, fn = [], []
    for _ in range(6000):
        x += 15.0e-3 / 6000.0
        fo.append(abs(old.force(x)))
        fn.append(abs(new.force(x)))
    fo, fn = np.asarray(fo), np.asarray(fn)
    assert old.punctured and new.punctured, "둘 다 관통해야 비교가 성립한다"
    assert np.max(np.abs(fn - fo)) < 1.2 * s59.F_SLIP, \
        f"삽입 중 차이가 {np.max(np.abs(fn - fo)):.3f} N — 미끄러짐 한계 안이어야 한다"


def test_the_new_switches_leave_the_earlier_experiments_untouched():
    a = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.10)
    b = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.10, breath_mm=0.0,
               retract_mm=0.0, master_lock=False, op_react_ms=0.0)
    assert a["e_min"] == b["e_min"] and a["final_depth_mm"] == b["final_depth_mm"]


# --------------------------------------------------------------------------- #
# 3) 정책 수준의 정직한 네거티브
# --------------------------------------------------------------------------- #
def test_retraction_reduces_the_load_but_costs_blind_motion():
    """B 절 — 후퇴는 사는 것이 있지만 그 대가가 더 크다(이 조직 파라미터에서)."""
    hold = s59.sweep(seeds=3, grip=True, retract_mm=0.0)
    ret = s59.sweep(seeds=3, grip=True, retract_mm=s59.RETRACT_MM)
    assert ret["df_held"] < hold["df_held"], "후퇴하면 조직에 얹는 몫이 줄어야 한다"
    assert ret["blind"] > 2 * hold["blind"], "그리고 맹행은 크게 늘어야 한다(정보 없는 움직임)"


def test_a_stickier_tissue_flips_the_retraction_verdict():
    """B 절의 '뒤집힐 조건' — 조직이 잘 안 놓아주면(미끄러짐 한계가 크면) 붙들기의 대가가 커진다.

    정책 결론이 **어느 조직인가**에 달려 있다는 주장의 근거다.
    """
    soft = s59.hold_and_breathe(True, breath_mm=5.0)[0]
    sticky_tissue = s59.GrippingTissue(f_slip=4.0)
    x = s59.X_SURFACE
    for _ in range(4000):
        x += 12.0e-3 / 4000.0
        sticky_tissue.force(x)
    f = [abs(sticky_tissue.force(x - 5.0e-3 * np.sin(2 * np.pi * s59.BREATH_HZ * k * jc.DT)))
         for k in range(4000)]
    sticky = max(f) - min(f)
    assert sticky > 1.7 * soft, f"끈적한 조직에서 붙들기의 부하가 커야 한다: {soft:.2f} → {sticky:.2f} N"


# --------------------------------------------------------------------------- #
# 4) 술자 쪽에서 부호가 뒤집힌다
# --------------------------------------------------------------------------- #
def test_locking_the_master_makes_resumption_worse_in_both_operator_models():
    """C 절의 핵심 — 잠금은 의도를 없애지 않고 손에 저장하며, 반응할 신호까지 가린다."""
    for react in (False, True):
        free = s59.sweep(seeds=3, grip=True, react=react, master_lock=False)
        lock = s59.sweep(seeds=3, grip=True, react=react, master_lock=True)
        assert lock["mism"] < free["mism"], "잠그면 어긋남은 줄어든다(당연)"
        assert lock["vres"] > free["vres"], \
            f"react={react}: 그런데 복귀는 더 빠르다 {free['vres']:.0f} → {lock['vres']:.0f} mm/s"


def test_a_reacting_operator_is_what_actually_reduces_the_lunge():
    """이득은 실재하지만 **보편적이지 않다** — 그래서 짝지은 통계로 못박는다.

    중앙값 대 중앙값은 두 분포가 오른쪽으로 치우쳐 있어 효과를 과장한다(exp 52 의 검출률↔AUROC 와
    같은 실수). 같은 시드끼리 비교해 '대부분의 시드에서 줄어든다'만 주장한다.
    """
    n = 12
    fx = np.array([s59.run(seed=s, grip=True, react=False)["resume_vmax_mms"]
                   for s in range(n)])
    rc = np.array([s59.run(seed=s, grip=True, react=True)["resume_vmax_mms"]
                   for s in range(n)])
    wins = int((rc < fx).sum())
    assert wins >= 8, f"개선된 시드 {wins}/{n}"
    assert np.median((fx - rc) / fx) > 0.10, \
        f"짝지은 감소율 중앙값 {np.median((fx - rc) / fx) * 100:.0f}%"


def test_the_reaction_rule_does_not_destabilise_the_human_loop():
    """exp 50 이 폐기한 '이득 있는 시각 폐루프'와 달리 목표를 얼려두는 규칙이라 발산하지 않는다."""
    s = s59.sweep(seeds=3, grip=True, react=True)
    assert s["n_div"] == 0
    assert np.isfinite(s["osc"]) and s["osc"] < 2.0, f"진동 {s['osc']:.2f} mm"


# --------------------------------------------------------------------------- #
# 합친 정책
# --------------------------------------------------------------------------- #
def test_the_recommended_combination_keeps_exp58_properties():
    """붙들기 + 술자 반응 — 맹행을 늘리지 않고 복귀만 줄이며, 수동성·완주 유지."""
    base = s59.sweep(seeds=3, grip=True)
    best = s59.sweep(seeds=3, grip=True, react=True)
    assert best["blind"] <= base["blind"] * 1.15
    assert best["vres"] < base["vres"]
    assert best["e_min"] >= -1e-12, "국소·소산 추가라 채널 장부를 건드리지 않아야 한다"
    assert best["final"] > 40.0


def test_run_is_reproducible():
    a = s59.run(seed=1, grip=True, react=True)
    b = s59.run(seed=1, grip=True, react=True)
    assert a["blind_max_mm"] == b["blind_max_mm"]
    assert a["resume_vmax_mms"] == b["resume_vmax_mms"]


def test_main_runs():
    out = s59.main(quick=True)
    assert set(out) == {"A", "B", "C", "D"}
    assert Path("assets/59_what_is_safe_state.png").exists()
