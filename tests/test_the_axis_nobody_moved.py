"""exp 68 의 주장을 회귀로 고정한다.

주장은 넷이다.
  1. **선불은 자기를 먹는다** — 적립할 여유가 없고, 일부러 적립하면 주입이 오히려 는다.
  2. **exp 67 의 '그런 칸은 없다'는 2 차원 절단면의 결과였다** — 세 번째 축에 칸이 있다.
  3. **그 칸은 표적을 지나치는데 R18 이 바닥이라 사슬의 어떤 검사도 못 본다.**
  4. **그리고 그 수동성은 채널 조건부다** — 최악 조건에서 무너지고 위해 축에서도 진다.

이 실험도 공용 시뮬레이터를 건드렸으므로(prepay_d, e_hand, e_ctrl 궤적 로깅) **앞 실험들의
숫자가 그대로인지**부터 고정한다. exp 66 이 그 자리에서 실제로 버그를 잡았고, exp 67 도 같은
자리를 첫 테스트로 뒀다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("68_the_axis_nobody_moved")
m67 = import_module("67_the_leak_is_the_task")
jc = import_module("56_jittery_channel")

SEEDS = 4


# --------------------------------------------------------------------------- #
# 0. 공용 시뮬레이터를 또 건드렸으니 앞 실험이 그대로인지부터
# --------------------------------------------------------------------------- #
def test_the_new_knobs_did_not_change_the_earlier_numbers():
    """prepay_d·e_hand·장부 궤적 로깅을 넣고도 exp 65~67 이 보고한 실행이 같아야 한다."""
    r = jc.run("tdpa", seed=0, jitter_ms=20.0)
    assert r["final_depth_mm"] == pytest.approx(50.76, abs=0.02)
    assert r["e_ctrl_max"] * 1e3 == pytest.approx(80.57, abs=0.05)
    assert r["e_min"] >= -1e-9


def test_the_logged_ledger_matches_the_scalar_it_replaced():
    """궤적 로깅이 진짜 그 장부인지 — 최댓값이 e_ctrl_max 와 같아야 한다."""
    r = jc.run("tdpa", seed=2, jitter_ms=20.0)
    assert max(r["log"]["e_ctrl"]) == pytest.approx(r["e_ctrl_max"], abs=1e-12)


# --------------------------------------------------------------------------- #
# 1. 선불은 자기를 먹는다
# --------------------------------------------------------------------------- #
def test_most_of_the_injection_is_spent_before_touching_anything():
    """**exp 67 의 '침투의 가격'을 시간축에서 다시 본다** — 상당 부분이 접촉 전에 쌓인다."""
    r = jc.run("tdpa", seed=0, jitter_ms=20.0)
    e = np.array(r["log"]["e_ctrl"])
    fe = np.abs(np.array(r["log"]["fe"]))
    touch = int(np.argmax(fe > 1e-3))
    assert touch > 0                                   # 자유공간 구간이 실제로 있고
    assert e[touch] * 1e3 > 10.0                       # 닿기 전에 이미 10 mJ 넘게 쌓인다


def test_there_is_no_reserve_to_save_up():
    """선불 설계의 전제가 무너지는 자리 — 장부가 음수로 내려가는 폭이 무의미하다."""
    r = jc.run("tdpa", seed=0, jitter_ms=20.0)
    e = np.array(r["log"]["e_ctrl"])
    assert min(e) * 1e3 > -1.0                         # 여유가 1 mJ 도 안 되는데
    assert max(e) * 1e3 > 50.0                         # 갚아야 할 것은 50 mJ 이 넘는다


def test_prepaying_makes_the_injection_worse():
    """**예측이 틀린 자리.** 갚을 재원을 만드는 행위가 빚을 같이 만든다."""
    base = m.clean(seeds=SEEDS)
    paid = m.clean(seeds=SEEDS, prepay_d=3000.0)
    assert paid["e_ctrl"] > base["e_ctrl"]             # 주입이 오히려 늘고
    assert paid["perr"] > base["perr"] * 3.0           # 손과 도구가 크게 어긋난다


def test_the_prepay_reserve_never_approaches_what_is_owed():
    """적립이 늘긴 하는데 규모가 안 맞는다 — 방향이 맞아도 자릿수가 틀리면 설계가 아니다."""
    paid = m.clean(seeds=SEEDS, prepay_d=3000.0)
    assert paid["reserve"] < -1e-3                     # 실제로 적립은 되지만
    assert abs(paid["reserve"]) * 1e3 < 0.5 * paid["e_ctrl"] * 1e3   # 빚의 절반도 못 된다


# --------------------------------------------------------------------------- #
# 2. 세 번째 축 — exp 67 의 결론은 2 차원 절단면이었다
# --------------------------------------------------------------------------- #
def test_raising_the_impedance_cuts_the_injection_while_keeping_the_task():
    """exp 67 은 이득을 **내려서** 비율을 맞췄고(과제 상실) 이쪽은 임피던스를 **올린다**."""
    low = m.clean(seeds=SEEDS)
    high = m.clean(seeds=SEEDS, b_wave=120.0)
    assert high["e_ctrl"] < low["e_ctrl"] * 0.5
    assert high["depth"] >= m.DEPTH_BAR


def test_raising_the_impedance_sells_transparency():
    """공짜가 아니다 — 힘 표시가 조직에서 멀어진다."""
    low = m.clean(seeds=SEEDS)
    high = m.clean(seeds=SEEDS, b_wave=120.0)
    assert high["ferr"] > low["ferr"] * 2.0


def test_a_passive_and_finishing_cell_exists_off_the_two_dimensional_slice():
    """**exp 67 의 헤드라인에 대한 정정.** 상자 안에는 있다."""
    cell = m.clean(seeds=SEEDS, **m.PASSIVE_CELL)
    assert cell["e_ctrl"] <= 1e-9
    assert cell["depth"] >= m.DEPTH_BAR


def test_the_slice_experiment_67_scanned_really_has_none():
    """정정이 정정이려면 **exp 67 이 본 면에는 없다**는 것도 같이 고정돼야 한다."""
    G = m67.scan(ds=(10.0, 30.0, 60.0, 150.0), lams=(0.0, 6.0, 24.0, 48.0), seeds=2)
    assert not any(v["e_ctrl"] <= 1e-9 and v["depth"] >= m67.DEPTH_BAR for v in G.values())


def test_the_passive_cell_needs_an_impedance_far_from_the_one_in_use():
    """싼 값에 오는 게 아니다 — exp 50 이 고른 값 근처에서는 안 온다."""
    near = m.clean(seeds=SEEDS, b_wave=4 * m.B_CHAIN, d_s=m.PASSIVE_CELL["d_s"],
                   lam_pos=m.PASSIVE_CELL["lam_pos"])
    assert near["e_ctrl"] > 1e-9
    assert m.PASSIVE_CELL["b_wave"] >= 10 * m.B_CHAIN


# --------------------------------------------------------------------------- #
# 3. 그 칸은 표적을 지나치고, 기준은 그걸 못 본다
# --------------------------------------------------------------------------- #
def test_the_passive_cell_overshoots_the_target():
    """**이 실험의 진짜 소득.** 완주는 하는데 표적을 지나친다."""
    cell = m.clean(seeds=SEEDS, **m.PASSIVE_CELL)
    assert cell["depth"] > m.TARGET_MM + 2.0


def test_the_completion_bar_is_a_floor_and_cannot_see_it():
    """**기준의 결함을 회귀로 박는다** — 초과하는 구성이 R18 을 통과한다.

    이 테스트가 실패로 바뀌는 날은 R18 이 창(window)으로 고쳐진 날이고, 그때 이 파일도
    같이 고쳐야 한다. 지금은 **통과한다는 사실 자체가 결함의 증거**다.
    """
    cell = m.clean(seeds=SEEDS, **m.PASSIVE_CELL)
    assert cell["depth"] >= m.DEPTH_BAR               # R18 을 통과하면서
    assert cell["depth"] > m.TARGET_MM                # 표적을 지나친다
    assert m.DEPTH_BAR < m.TARGET_MM                  # 기준이 표적 아래의 바닥이라서


def test_the_baseline_undershoots_the_same_target():
    """대조 — 사슬은 반대쪽으로 못 미친다. 표적이 창이었다면 둘 다 걸렸을 것이다."""
    base = m.clean(seeds=SEEDS)
    assert base["depth"] < m.TARGET_MM


# --------------------------------------------------------------------------- #
# 4. 대가와 조건부성
# --------------------------------------------------------------------------- #
def test_the_bill_goes_to_the_operator():
    """**청구서가 누구에게 가는가.** 술자의 손이 하는 일이 자릿수로 는다."""
    base = m.clean(seeds=SEEDS)
    cell = m.clean(seeds=SEEDS, **m.PASSIVE_CELL)
    assert cell["hand"] > base["hand"] * 2.0


def test_the_two_prescriptions_sell_different_axes():
    """exp 66 의 관측기와 이 칸은 **서로 다른 축을 판다** — 하나를 다른 하나로 못 바꾼다."""
    cell = m.clean(seeds=SEEDS, **m.PASSIVE_CELL)
    obs = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True)
    assert cell["ferr"] > obs["ferr"] * 3.0            # 칸은 힘 투명성을 팔고
    assert obs["perr"] > cell["perr"] * 3.0            # 관측기는 위치 추종을 판다


def test_the_passive_cell_collapses_in_the_worst_condition():
    """**그 수동성은 채널 조건부다** — exp 66 이 시드 수로 배운 것의 채널판이다."""
    hc = m.harsh(seeds=m.HARM_SEEDS, **m.PASSIVE_CELL)
    ho = m.harsh(seeds=m.HARM_SEEDS, drift_mode="po", po_strict=True)
    assert hc["bad"]                                   # 위반 시드가 생기고
    assert len(hc["bad"]) > len(ho["bad"])             # 관측기보다 많이 깨진다


def test_the_passive_cell_also_loses_on_the_harm_axis():
    """수동성을 샀다고 위해가 좋아지는 게 아니다 — exp 63 축에서 오히려 나빠진다."""
    hb = m.harsh(seeds=m.HARM_SEEDS)
    hc = m.harsh(seeds=m.HARM_SEEDS, **m.PASSIVE_CELL)
    assert hc["drag"] > hb["drag"] * 1.2


def test_main_runs():
    """quick 모드로 전 절이 끝까지 도는지 — 표·그림까지 포함한 연기 시험."""
    out = m.main(quick=True)
    assert set(out) == {"A", "B", "G", "D", "F", "both", "target"}
    assert Path("assets/68_the_axis_nobody_moved.png").exists()
