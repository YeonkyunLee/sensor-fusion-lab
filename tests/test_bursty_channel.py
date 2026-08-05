"""exp 57 — 연집 손실과 긴 지연 꼬리.

이 실험의 결론은 대부분 **정직한 네거티브**(예측이 틀렸다)라서, 테스트도 두 가지를 지킨다.
  1) 채널 모형이 정의대로인가 — 평균 손실률·평균 지연을 정말 고정하고 모양만 바꾸는가.
     이게 깨지면 '연집성의 효과'가 아니라 '늘어난 지연/손실의 효과'를 재게 된다.
  2) 주장한 인과가 그 원인에서 오는가 — 특히 "보증을 깨는 항이 브레이크였다"는 뒤집기.
"""

from importlib import import_module
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
bc = import_module("57_bursty_channel")


# --------------------------------------------------------------------------- #
# 채널 모형이 정의대로인가 (공정한 비교의 전제)
# --------------------------------------------------------------------------- #
def test_gilbert_elliott_holds_the_average_loss_rate_across_burst_lengths():
    """연집 길이를 바꿔도 **평균 손실률이 같아야** 모양만 바꾼 비교가 된다.

    긴 연집은 사건 수가 적어 한 실행의 실현값이 크게 흔들린다(실험의 한계로 기록해 둔 사항).
    그래서 여러 시드를 모아 **정상상태** 값이 맞는지를 본다.
    """
    for L in (1.0, 10.0, 80.0):
        lost = sent = 0
        for seed in range(8):
            ch = bc.BurstyChannel(np.random.default_rng(seed), loss=0.10, burst_len=L)
            for k in range(40000):
                ch.send(k, 0.0)
            lost += ch.n_lost
            sent += ch.n_sent
        assert abs(lost / sent - 0.10) < 0.015, f"L={L}: 손실률 {lost / sent:.3f}"


def test_longer_bursts_make_longer_outages_at_the_same_loss_rate():
    ch1 = bc.BurstyChannel(np.random.default_rng(1), loss=0.10, burst_len=1.0)
    ch2 = bc.BurstyChannel(np.random.default_rng(1), loss=0.10, burst_len=80.0)

    def longest(ch):
        best = cur = 0
        for k in range(20000):
            n0 = ch.n_lost
            ch.send(k, 0.0)
            cur = cur + 1 if ch.n_lost > n0 else 0
            best = max(best, cur)
        return best

    assert longest(ch2) > 10 * longest(ch1)


def test_matching_the_mean_delay_is_what_makes_the_comparison_fair():
    """꼬리를 붙이면 평균 지연이 오른다. 공칭을 깎아 **평균**을 맞춘 뒤 비교해야 한다."""
    r0 = bc.run("zoh", seed=0)
    r1 = bc.run("zoh", seed=0, tail_ms=bc.TAIL_MS)
    assert abs(r0["mean_delay_ms"] - r1["mean_delay_ms"]) < 1e-9
    assert not r1["mean_clipped"]
    # 맞추지 않으면 평균이 꼬리 평균만큼 올라간다 = 다른 실험이 된다
    r2 = bc.run("zoh", seed=0, tail_ms=bc.TAIL_MS, match_mean=False)
    assert r2["mean_delay_ms"] > r1["mean_delay_ms"] + 5.0


def test_pareto_tail_has_no_practical_maximum():
    """C 절의 전제 — α<2 라 분산이 무한하고, 분위수가 급격히 벌어진다."""
    assert bc.TAIL_ALPHA < 2.0
    assert (bc.tail_quantile_ms(bc.TAIL_MS, 0.999)
            > 3 * bc.tail_quantile_ms(bc.TAIL_MS, 0.99))


def test_startup_is_not_counted_as_an_outage():
    """첫 수신 전 구간은 공칭 지연이지 사건이 아니다 — 안 빼면 모든 조건이 50 ms 로 깔린다."""
    r = bc.run("zoh", seed=0, loss=0.10, burst_len=1.0)
    assert r["max_starve_ms"] < 20.0, f"{r['max_starve_ms']:.0f} ms"


# --------------------------------------------------------------------------- #
# 주장한 인과
# --------------------------------------------------------------------------- #
def test_holding_a_stale_command_is_self_limiting():
    """B 절 — 낡은 명령에도 정지 평형이 있어 팔이 수렴한다. 이게 exp 56 결론이 버틴 이유다.

    배율은 시드에 따라 1.7~3.3배로 흔들린다(연집이 길면 사건 수가 적다). 방향만 못박는다.
    """
    s = bc.sweep("zoh", seeds=4, tail_ms=bc.TAIL_MS, loss=0.10, burst_len=80.0)
    assert np.isfinite(s["hold_late_um"]), "그만큼 긴 홀드가 있어야 이 비교가 성립한다"
    assert s["hold_late_um"] < 0.8 * s["hold_early_um"], \
        f"early {s['hold_early_um']:.1f} vs late {s['hold_late_um']:.1f} um"


def test_clumping_does_not_increase_total_blind_travel():
    """정직한 네거티브 — 평균 손실률이 같으면 총량은 안 늘고 오히려 줄어든다."""
    a = bc.sweep("zoh", seeds=4, loss=0.10, burst_len=1.0)
    b = bc.sweep("zoh", seeds=4, loss=0.10, burst_len=80.0)
    assert b["blind_mm"] <= a["blind_mm"]
    # 대신 한 사건은 커진다
    assert b["max_starve_ms"] > 10 * a["max_starve_ms"]


def test_e_min_cannot_rank_the_conditions_but_drawdown_can():
    """A 절의 지표 선택. E_min 은 선로 위 저수지에 가려 순위를 못 만든다."""
    mild = bc.sweep("zoh", seeds=4, tail_ms=bc.TAIL_MS)
    worst = bc.sweep("zoh", seeds=4, tail_ms=bc.TAIL_MS, loss=0.10,
                     burst_len=bc.BURST_LEN)
    assert worst["e_drawdown"] > mild["e_drawdown"], "최대 낙폭은 정렬해야 한다"


def test_gating_the_drift_term_makes_blind_travel_worse():
    """E 절의 뒤집기 — 보증 밖의 그 항이 낡은 setpoint 를 향한 **브레이크**였다."""
    kw = dict(tail_ms=bc.TAIL_MS, loss=0.10, burst_len=160.0)
    bud = bc.sweep("tdpa", seeds=4, **kw)
    gated = bc.sweep("tdpa", seeds=4, lam_gate=True, **kw)
    assert gated["blind_max_mm"] > bud["blind_max_mm"], \
        f"예산 {bud['blind_max_mm']:.2f} vs 게이트 {gated['blind_max_mm']:.2f} mm"


def test_the_budget_still_restores_passivity_under_bursts():
    """D 절 — 예산은 연집에서도 구성상 수동적이다(가동률만 올라간다)."""
    z = bc.sweep("zoh", seeds=4, tail_ms=bc.TAIL_MS, loss=0.10, burst_len=bc.BURST_LEN)
    t = bc.sweep("tdpa", seeds=4, tail_ms=bc.TAIL_MS, loss=0.10, burst_len=bc.BURST_LEN)
    assert z["e_min"] < -1e-9, "ZOH 는 연집에서 능동이어야 한다"
    assert t["e_min"] >= -1e-12
    assert t["final_depth_mm"] > 0.9 * z["final_depth_mm"], "과제는 계속 완주해야 한다"


def test_an_undersized_playout_buffer_turns_delay_into_loss():
    """C 절 — 재생 기한이 없던 손실을 만든다. 망 자체의 손실률은 그대로다."""
    none = bc.sweep("zoh", seeds=3, tail_ms=bc.TAIL_MS, loss=0.05,
                    burst_len=bc.BURST_LEN, buf_ms=0.0)
    small = bc.sweep("zoh", seeds=3, tail_ms=bc.TAIL_MS, loss=0.05,
                     burst_len=bc.BURST_LEN,
                     buf_ms=bc.tail_quantile_ms(bc.TAIL_MS, 0.5))
    assert none["late_frac"] == 0.0, "기한이 없으면 '늦은 패킷' 이라는 범주가 없다"
    assert small["late_frac"] > 0.2
    assert small["osc_mm"] >= none["osc_mm"], "게다가 지연도 이미 더해져 있다"


def test_a_large_buffer_buys_passivity_with_latency_that_costs_oscillation():
    small = bc.sweep("zoh", seeds=3, tail_ms=bc.TAIL_MS, loss=0.05,
                     burst_len=bc.BURST_LEN, buf_ms=0.0)
    big = bc.sweep("zoh", seeds=3, tail_ms=bc.TAIL_MS, loss=0.05,
                   burst_len=bc.BURST_LEN,
                   buf_ms=bc.tail_quantile_ms(bc.TAIL_MS, 0.999))
    assert big["late_frac"] < 0.01
    assert big["osc_mm"] > 3 * small["osc_mm"]


def test_run_is_reproducible():
    a = bc.run("tdpa", seed=2, tail_ms=bc.TAIL_MS, loss=0.10, burst_len=40.0)
    b = bc.run("tdpa", seed=2, tail_ms=bc.TAIL_MS, loss=0.10, burst_len=40.0)
    assert a["e_min"] == b["e_min"] and a["blind_max_mm"] == b["blind_max_mm"]


def test_exp56_still_passes_its_own_conditions_through_the_new_seam():
    """리팩터링 확인 — 채널을 끼울 수 있게 바꿨어도 기본 경로는 exp 56 그대로여야 한다."""
    jc = import_module("56_jittery_channel")
    a = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.10)
    b = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.10, chan=jc.Channel)
    assert a["e_min"] == b["e_min"] and a["final_depth_mm"] == b["final_depth_mm"]


def test_main_runs():
    out = bc.main(quick=True)
    assert set(out) == {"A", "B", "C", "D", "E"}
    assert Path("assets/57_bursty_channel.png").exists()
