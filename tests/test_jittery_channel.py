"""exp 56 — 지터·손실 채널과 시간영역 수동성 제어.

핵심은 두 갈래다.
  1) 채널 장부가 **정의대로** 동작하는가(수동성 판정을 여기에 의지하므로).
  2) 실험이 주장한 인과가 실제로 그 원인에서 오는가 — 특히 "가정을 걷어냈는데 아무 일도
     없었던" 이유가 **시험이 여기를 못 한 것**이라는 주장.
"""

from importlib import import_module
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
jc = import_module("56_jittery_channel")


# --------------------------------------------------------------------------- #
# 채널 자체
# --------------------------------------------------------------------------- #
def test_channel_delivers_after_the_nominal_delay():
    ch = jc.Channel(np.random.default_rng(0), delay_ms=10.0)
    ch.send(0, (1.0, 0.0, 0.0))
    n = int(round(10e-3 / jc.DT))
    for k in range(n):
        _, fresh = ch.recv(k)
        assert not fresh, f"스텝 {k} 에서 지연보다 일찍 도착했다"
    payload, fresh = ch.recv(n)
    assert fresh and payload[0] == 1.0


def test_reordered_stale_packets_are_discarded():
    """지연이 줄어드는 지터는 낡은 패킷을 만든다. 그걸 쓰면 시간이 거꾸로 간다."""
    ch = jc.Channel(np.random.default_rng(0), delay_ms=10.0)
    n = int(round(10e-3 / jc.DT))
    ch.q = [(n, 0, (1.0, 0.0, 0.0)), (n, 5, (2.0, 0.0, 0.0))]   # 같은 스텝에 두 개 도착
    payload, fresh = ch.recv(n)
    assert fresh and payload[0] == 2.0, "더 나중에 보낸 것을 채택해야 한다"
    assert ch.n_stale == 1


def test_holding_repeats_the_last_sample_and_zero_fill_does_not():
    ch = jc.Channel(np.random.default_rng(0), delay_ms=1.0, zero=(0.0, 0.0, 0.0))
    ch.send(0, (7.0, 0.0, 0.0))
    ch.recv(1)
    assert ch.recv(2, hold=True)[0][0] == 7.0
    assert ch.recv(3, hold=False)[0][0] == 0.0


def test_dejitter_buffer_removes_starvation_at_the_cost_of_latency():
    """버퍼는 지터를 **추가 상수 지연**으로 바꾼다 — 굶는 스텝이 사라져야 한다."""
    a = jc.run("zoh", seed=0, jitter_ms=20.0)
    b = jc.run("zoh", seed=0, jitter_ms=20.0, buf_ms=30.0)
    assert a["starve_frac"] > 0.5, "지터 ±20 ms 면 굶는 스텝이 많아야 한다"
    assert b["starve_frac"] < 0.2 * a["starve_frac"]


# --------------------------------------------------------------------------- #
# 에너지 장부 — 수동성 판정의 근거
# --------------------------------------------------------------------------- #
def test_constant_delay_channel_is_passive():
    """상수 지연·무손실이면 채널 에너지는 '선로 위에 떠 있는 에너지'라 음수가 될 수 없다."""
    r = jc.run("zoh", seed=0)
    assert r["e_min"] >= -1e-12, f"E_min = {r['e_min']:.3e} J"


def test_energy_budget_never_lets_the_channel_create_energy():
    """TDPA 의 성립 조건 — 구성상 소비 ≤ 통보된 예산이므로 어떤 지터·손실에서도 수동."""
    for jit, loss in ((20.0, 0.0), (0.0, 0.30), (40.0, 0.30)):
        r = jc.run("tdpa", seed=1, jitter_ms=jit, loss=loss)
        assert r["e_min"] >= -1e-12, f"지터 {jit}, 손실 {loss}: E_min={r['e_min']:.3e}"


def test_zero_fill_is_passive_but_loses_the_drive():
    """0 채움은 안전한 쪽 극단이다: 에너지를 안 만드는 대신 도구가 못 나간다."""
    z = jc.run("zero", seed=0, jitter_ms=20.0, loss=0.20)
    h = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.20)
    assert z["e_min"] >= -1e-12
    assert z["final_depth_mm"] < 0.5 * h["final_depth_mm"]


def test_the_system_level_balance_cannot_see_the_violation():
    """국소 소산이 섞인 지표로는 채널 수동성을 판정할 수 없다 — 관측기 배치 규칙."""
    r = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.20)
    assert r["e_min"] < -1e-9, "이 조건에서는 파동 채널이 능동이어야 한다"
    assert r["e_sys_min"] >= -1e-12, "그런데 시스템 수지는 위반을 못 본다"


# --------------------------------------------------------------------------- #
# 실험이 주장한 인과
# --------------------------------------------------------------------------- #
def test_exp50_gain_cannot_reach_the_target_so_the_channel_is_barely_excited():
    """A 절의 '아무 일도 없음'은 채널이 튼튼해서가 아니라 **시험이 약해서**다."""
    r = jc.run("zoh", seed=0, lam_pos=jc.LAM_50)
    assert r["final_depth_mm"] < 0.75 * jc.X_TARGET * 1e3, \
        "exp 50 이득으로는 표적에 한참 못 미쳐야 한다(정상상태 오차)"
    assert r["punct_ms"] != r["punct_ms"], "관통도 일어나지 않는다(NaN)"


def test_completing_the_task_makes_the_same_jitter_create_far_more_energy():
    """B 절 — 결함은 내내 있었고 크기가 신호에 비례할 뿐이다.

    생성은 **굶는 순간이 어디에 떨어지느냐**에 달려 최악값으로 봐야 한다(한 시드는 0 이 나온다).
    """
    lo = jc.sweep("zoh", seeds=4, jitter_ms=20.0, lam_pos=jc.LAM_50)
    hi = jc.sweep("zoh", seeds=4, jitter_ms=20.0, lam_pos=jc.LAM_TASK)
    assert hi["final_depth_mm"] > lo["final_depth_mm"] + 10.0
    assert -hi["e_min"] > 20 * (-lo["e_min"]), \
        f"lo={lo['e_min']:.3e} hi={hi['e_min']:.3e}"


def test_the_buffer_buys_passivity_and_sells_performance():
    """C 절의 정직한 네거티브 — 표준 처방이 성능을 깎는다."""
    a = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.05)
    b = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.05, buf_ms=45.0)
    assert a["e_min"] < -1e-9 and b["e_min"] >= -1e-12
    assert b["osc_mm"] > 2 * a["osc_mm"]
    assert b["pos_err_mm"] > a["pos_err_mm"]


def test_energy_budget_beats_the_buffer_on_both_counts():
    """D 절의 구성적 결론: 예산은 지연을 안 내고도 수동성을 되찾는다."""
    buf = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.20, buf_ms=30.0)
    bud = jc.run("tdpa", seed=0, jitter_ms=20.0, loss=0.20)
    assert bud["e_min"] >= -1e-12
    assert bud["osc_mm"] < buf["osc_mm"]
    assert bud["final_depth_mm"] > buf["final_depth_mm"] - 1.0


def test_cumulative_energy_is_loss_tolerant_and_increments_are_not():
    """E 절 — 같은 알고리즘, 다른 페이로드. 단조량이라 손실이 복구된다."""
    cum = jc.run("tdpa", seed=0, jitter_ms=20.0, loss=0.30, energy_mode="cumulative")
    inc = jc.run("tdpa", seed=0, jitter_ms=20.0, loss=0.30, energy_mode="increment")
    assert inc["att_duty"] > 5 * cum["att_duty"]
    assert cum["final_depth_mm"] > inc["final_depth_mm"] + 5.0


def test_the_link_is_oversampled_for_this_task():
    """A 절 — 1 kHz 는 국소 루프의 요구이지 채널이 날라야 할 정보량이 아니다."""
    fast = jc.run("zoh", seed=0, rate_hz=1000)
    slow = jc.run("zoh", seed=0, rate_hz=50)
    assert abs(slow["final_depth_mm"] - fast["final_depth_mm"]) < 2.0
    assert abs(slow["force_err_punc_N"] - fast["force_err_punc_N"]) < 0.3


def test_the_wall_protects_the_hand_not_the_tool():
    """G 절 — 안전 지표를 위해가 발생하는 지점에서 재야 한다(exp 41 의 FRE≠TRE)."""
    r = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.05, vf_stiffness=12000.0)
    assert r["master_pen_mm"] > 0.05, "손은 벽에 닿아 있어야 한다"
    assert r["lag_mm"] > 5 * r["master_pen_mm"], \
        "도구 뒤처짐이 마스터 침범보다 훨씬 크다 = 마스터 숫자로는 알 수 없다"


def test_raising_the_wave_impedance_does_not_fix_jitter():
    """F 절 — 대책은 자기가 겨냥한 오차원만 고친다(exp 53)."""
    nom = jc.run("zoh", seed=0, jitter_ms=20.0, loss=0.20)
    big = jc.run("bigb", seed=0, jitter_ms=20.0, loss=0.20)
    assert big["e_min"] < -1e-9, "b 를 키워도 채널은 여전히 능동이다"
    assert big["force_err_N"] > nom["force_err_N"], "그리고 투명성은 더 낸다"


def test_run_is_reproducible():
    a = jc.run("tdpa", seed=3, jitter_ms=20.0, loss=0.20)
    b = jc.run("tdpa", seed=3, jitter_ms=20.0, loss=0.20)
    assert a["e_min"] == b["e_min"] and a["final_depth_mm"] == b["final_depth_mm"]


def test_main_runs():
    out = jc.main(quick=True)
    assert set(out) == {"A", "A2", "R", "B", "C", "D", "M", "F", "G"}
    assert Path("assets/56_jittery_channel.png").exists()
