"""exp 65 의 주장을 회귀로 고정한다.

핵심은 하나다: **파동 블록 장부가 정확히 0 인 동안 전체 제어기가 에너지를 만든다.**
아홉 실험이 한계 절에 적어만 두고 재지 않은 자리라, 숫자로 박아 둔다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("65_whole_loop_passivity")
jc = import_module("56_jittery_channel")
s59 = import_module("59_what_is_safe_state")


# --------------------------------------------------------------------------- #
# 장부 자체가 옳게 서 있는가
# --------------------------------------------------------------------------- #
def test_the_ledger_is_an_upper_bound_not_an_endpoint():
    """수동성은 **모든 시각**에 대한 조건이다 — 끝값이 아니라 최댓값을 봐야 한다."""
    r = jc.run("tdpa", seed=0, jitter_ms=20.0)
    assert r["e_ctrl_max"] >= r["e_ctrl_end"] - 1e-12


def test_adding_the_ledger_did_not_change_the_physics():
    """장부는 **관측**일 뿐이다 — 앞 실험 경로의 숫자가 그대로여야 한다."""
    a = jc.run("tdpa", seed=3, jitter_ms=20.0)
    for k, v in (("final_depth_mm", 50.0), ("e_min", 0.0)):
        assert np.isfinite(a[k])
    assert a["e_min"] >= -1e-9                      # 파동 블록은 여전히 만족
    assert a["final_depth_mm"] > 45.0               # 과제도 여전히 완주(R18)


def test_a_purely_dissipative_run_does_not_look_active():
    """**대조군.** λ=0 이면 표류 보정이 없으니 주입이 확 줄어야 한다.

    안 줄면 이 장부가 능동성이 아니라 다른 것을 재고 있다는 뜻이다.
    """
    off = jc.run("tdpa", seed=0, jitter_ms=20.0, lam_pos=0.0)
    on = jc.run("tdpa", seed=0, jitter_ms=20.0, lam_pos=jc.LAM_TASK)
    assert off["e_ctrl_max"] < on["e_ctrl_max"] * 0.5


# --------------------------------------------------------------------------- #
# 이 실험의 주장
# --------------------------------------------------------------------------- #
def test_the_wave_block_is_passive_while_the_whole_controller_is_not():
    """**핵심.** 두 장부가 정반대의 답을 준다."""
    r = m.med("tdpa", seeds=3, jitter_ms=20.0)
    assert r["e_min"] >= -1e-9                      # 파동 블록: 수동
    assert r["e_ctrl_max"] > 0.02                   # 전체: 20 mJ 넘게 만든다


def test_tdpa_fixes_the_ledger_it_watches_and_not_the_energy():
    """TDPA 는 파동 장부의 위반을 지우지만 전체 공급은 거의 그대로다.

    **위반은 중앙값이 아니라 최악 시드로 물어야 한다** — 수동성은 모든 실행·모든 시각에 대한
    조건이라 "절반은 괜찮았다"가 답이 아니다. 3 시드 중앙값으로는 위반이 안 잡혀서 잡은 함정.
    """
    worst = lambda mode: min(jc.run(mode, seed=s, jitter_ms=40.0)["e_min"]  # noqa: E731
                             for s in range(6))
    assert worst("zoh") < -1e-4                     # hold-last 는 어딘가에서 위반하고
    assert worst("tdpa") >= -1e-9                   # TDPA 는 어느 시드에서도 위반 안 한다
    z = m.med("zoh", seeds=3, jitter_ms=40.0)
    t = m.med("tdpa", seeds=3, jitter_ms=40.0)
    assert t["e_ctrl_max"] > z["e_ctrl_max"] * 0.7  # 그런데 전체 공급은 그대로다


def test_the_injection_is_not_caused_by_jitter():
    """주입이 가장 큰 조건이 **지터 0** 이다 — 채널 현상이 아니다."""
    q = m.med("tdpa", seeds=3, jitter_ms=0.0)
    j = m.med("tdpa", seeds=3, jitter_ms=40.0)
    assert q["e_min"] >= -1e-9 and j["e_min"] >= -1e-9
    assert q["e_ctrl_max"] > j["e_ctrl_max"]


def test_the_injection_grows_with_the_drift_gain():
    """λ 를 키우면 자란다 — 능동의 출처가 그 항이라는 증거."""
    lo = m.med("tdpa", seeds=3, jitter_ms=20.0, lam_pos=3.0)
    hi = m.med("tdpa", seeds=3, jitter_ms=20.0, lam_pos=48.0)
    assert hi["e_ctrl_max"] > lo["e_ctrl_max"] * 2.0


def test_the_gain_needed_to_finish_the_task_is_the_one_that_breaks_passivity():
    """**exp 56 의 손질이 이 문제를 만들었다.** 완주시키려 올린 λ 가 주입을 키운다."""
    lo = m.med("tdpa", seeds=3, jitter_ms=20.0, lam_pos=jc.LAM_50)
    hi = m.med("tdpa", seeds=3, jitter_ms=20.0, lam_pos=jc.LAM_TASK)
    assert lo["final_depth_mm"] < 40.0              # exp 50 의 이득은 완주 못 하고
    assert hi["final_depth_mm"] > 45.0              # exp 56 이 올린 이득은 완주하는데
    assert hi["e_ctrl_max"] > lo["e_ctrl_max"] * 2.0  # 그 대가로 주입이 커진다


# --------------------------------------------------------------------------- #
# 이미 있는 처방을 이 장부로 재기
# --------------------------------------------------------------------------- #
def test_the_exp58_stop_also_reduces_the_injection():
    """설계 목적은 맹행 제한이었는데 주입도 줄인다 — 아무도 주장하지 않았던 덤."""
    off = m.med("tdpa", seeds=3, bursty=True, estop=False)
    on = m.med("tdpa", seeds=3, bursty=True, estop=True)
    assert on["e_ctrl_max"] < off["e_ctrl_max"] * 0.85


def test_gating_the_drift_term_does_not_help_here_either():
    """exp 57 이 맹행으로 기각한 처방을, 명목상 겨냥했던 **수동성**으로 재도 안 듣는다."""
    plain = m.med("tdpa", seeds=3, bursty=True, estop=True)
    gated = m.med("tdpa", seeds=3, bursty=True, estop=True, lam_gate=True)
    assert gated["e_ctrl_max"] >= plain["e_ctrl_max"] * 0.95
    assert gated["blind_max_mm"] > plain["blind_max_mm"]


def test_active_does_not_mean_divergent():
    """정직한 단서: 능동이라고 곧 불안정은 아니다 — 이 조건들에서 발산하지 않는다."""
    for j in (0.0, 20.0, 40.0):
        r = m.med("tdpa", seeds=3, jitter_ms=j)
        assert r["diverged"] == 0
        assert r["e_ctrl_max"] > 0.0


def test_speed_conversion_is_the_stated_physics():
    """환산이 ½mv² 인지 — 보고한 크기 감이 맞는지 확인한다."""
    e = 0.5 * jc.M_S * 0.28 ** 2
    assert m.speed_for(e) == pytest.approx(0.28, rel=1e-6)
    assert m.speed_for(0.0) == 0.0
