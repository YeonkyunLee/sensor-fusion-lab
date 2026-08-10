"""exp 67 의 주장을 회귀로 고정한다.

주장은 넷이다.
  1. **남은 잔차는 구현 결함이 아니다** — 결합 이득을 파동 임피던스에 맞추면 정확히 수동이 된다.
  2. **그 대신 과제를 잃는다** — 그리고 두 이득의 격자 어디에도 '수동이면서 완주'는 없다.
  3. **주입은 침투의 가격이다** — 격자 위에서 깊이와 주입의 순위상관이 높다.
  4. **그래서 exp 66 의 관측기는 곡선 밖의 점이다** — 이득 두 개로는 못 가는 칸에 있다.

그리고 이 실험도 공용 시뮬레이터(exp 56 의 run)를 건드렸으므로 — d_s 인자, port_mode,
포트 잔차 계측 — **앞 실험들의 숫자가 그대로인지**부터 고정한다. exp 66 이 그 자리에서 실제로
버그를 하나 잡았고(반사파 송신 한 줄), 같은 자리가 이번에도 첫 테스트다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("67_the_leak_is_the_task")
jc = import_module("56_jittery_channel")

SEEDS = 4


# --------------------------------------------------------------------------- #
# 0. 공용 시뮬레이터를 또 건드렸으니 앞 실험이 그대로인지부터
# --------------------------------------------------------------------------- #
def test_the_new_knobs_did_not_change_the_earlier_numbers():
    """d_s 인자·port_mode·포트 계측을 넣고도 exp 65·66 이 보고한 실행이 숫자까지 같아야 한다."""
    r = jc.run("tdpa", seed=0, jitter_ms=20.0)
    assert r["final_depth_mm"] == pytest.approx(50.76, abs=0.02)
    assert r["e_ctrl_max"] * 1e3 == pytest.approx(80.57, abs=0.05)
    assert r["e_min"] >= -1e-9


def test_the_gain_argument_defaults_to_the_module_constant():
    """d_s 를 안 주면 지금까지의 D_S 와 **비트 단위로** 같아야 한다 — 아니면 위 회귀가 무의미하다."""
    a = jc.run("tdpa", seed=1, jitter_ms=20.0)
    b = jc.run("tdpa", seed=1, jitter_ms=20.0, d_s=jc.D_S)
    assert a["e_ctrl_max"] == b["e_ctrl_max"]
    assert a["final_depth_mm"] == b["final_depth_mm"]


# --------------------------------------------------------------------------- #
# 1. 잔차는 지목된 항도 채널도 아니다
# --------------------------------------------------------------------------- #
def test_the_residual_survives_a_perfect_channel_and_no_drift_term():
    """**exp 66 이 남긴 자리.** λ 를 문자 그대로 0 으로 두고 지터·손실도 0 으로 해도 남는다.

    exp 66 은 탱크 용량 0 을 '항을 완전히 껐다'로 적었는데 그건 **주입 방향만** 막는다.
    이것이 진짜 절제이고, 답은 같은 곳을 가리킨다.
    """
    r = jc.run("tdpa", seed=0, jitter_ms=0.0, lam_pos=0.0)
    assert r["e_ctrl_max"] * 1e3 > 10.0        # 여전히 만든다
    assert r["e_min"] >= -1e-9                 # 그동안 보던 파동 장부는 그 내내 0


# --------------------------------------------------------------------------- #
# 2. 이득을 임피던스에 맞추면 수동이 된다 — 그리고 과제를 잃는다
# --------------------------------------------------------------------------- #
def test_matching_the_gain_to_the_wave_impedance_is_exactly_passive():
    """**핵심 결과.** exp 66 은 이걸 '파동 변환을 다시 짜는 일'로 넘겼는데 이득 하나였다."""
    matched = m.clean(seeds=SEEDS, lam_pos=0.0, d_s=m.B_WAVE)
    assert matched["e_ctrl"] <= 1e-9


def test_but_the_matched_gain_cannot_do_the_task():
    """**실패할 수 없는 시험을 만들지 않기 위한 대조**(R18). 수동이지만 도구가 못 들어간다."""
    matched = m.clean(seeds=SEEDS, lam_pos=0.0, d_s=m.B_WAVE)
    assert matched["depth"] < m.DEPTH_BAR - 10.0


def test_the_injection_grows_with_the_coupling_gain():
    """사다리가 단조로워야 '이득이 값이다'라고 말할 수 있다."""
    ladder = [m.clean(seeds=SEEDS, lam_pos=0.0, d_s=g)["e_ctrl"]
              for g in (m.B_WAVE, 2 * m.B_WAVE, 4 * m.B_WAVE, 6 * m.B_WAVE)]
    assert all(b >= a for a, b in zip(ladder, ladder[1:]))
    assert ladder[-1] * 1e3 > 10.0


# --------------------------------------------------------------------------- #
# 3. 포트를 대수적으로 닫아도 안 산다 (예측 1 이 절반 틀렸다)
# --------------------------------------------------------------------------- #
def test_closing_the_port_algebraically_does_not_buy_passivity():
    """**예측이 틀린 자리.** 항등식을 구성상 참으로 만드는 것과 계가 수동인 것은 다르다."""
    legacy = m.clean(seeds=SEEDS, lam_pos=0.0)
    reflect = m.clean(seeds=SEEDS, lam_pos=0.0, port_mode="reflect")
    assert reflect["e_ctrl"] > 1e-9                  # 여전히 수동이 아니고
    assert reflect["e_ctrl"] > legacy["e_ctrl"]      # 오히려 나빠진다


def test_closing_the_port_sells_transparency():
    """예측의 **맞은 절반** — 손에 표시되는 힘이 조직에서 멀어진다."""
    legacy = m.clean(seeds=SEEDS)
    reflect = m.clean(seeds=SEEDS, port_mode="reflect")
    assert reflect["ferr"] > legacy["ferr"] * 3.0


def test_the_port_identity_residual_is_not_the_mechanism():
    """**내가 만든 계측기에 대한 자기 반증.**

    이 실험을 시작하게 만든 양이 실재하지만, 장부가 정확히 0 인 구성에서도 남는다.
    exp 66 의 '분해는 흐름이지 인과가 아니다' 가 한 층 아래에서 반복된다.
    """
    r = jc.run("tdpa", seed=0, jitter_ms=20.0, lam_pos=0.0, d_s=m.B_WAVE / 2)
    assert r["e_ctrl_max"] <= 1e-9                   # 장부는 위반이 없는데
    assert abs(r["e_port"]) * 1e3 > 1.0              # 포트 잔차는 남아 있다


# --------------------------------------------------------------------------- #
# 4. 격자 — 수동이면서 완주하는 칸은 없다
# --------------------------------------------------------------------------- #
def test_no_cell_is_both_passive_and_finishing():
    """**두 손잡이를 같이 움직여야만 물을 수 있는 질문.** exp 50~66 은 매번 하나만 쓸었다."""
    G = m.scan(ds=(10.0, 30.0, 60.0, 150.0), lams=(0.0, 6.0, 24.0, 48.0), seeds=2)
    both = [k for k, v in G.items() if v["e_ctrl"] <= 1e-9 and v["depth"] >= m.DEPTH_BAR]
    assert both == []
    assert any(v["e_ctrl"] <= 1e-9 for v in G.values())          # 수동인 칸은 있고
    assert any(v["depth"] >= m.DEPTH_BAR for v in G.values())    # 완주하는 칸도 있다


def test_the_injection_is_the_price_of_penetrating():
    """격자 위에서 깊이와 주입이 같이 간다 — 어느 항의 결함이 아니라 **가격표**라는 근거."""
    G = m.scan(ds=(10.0, 30.0, 60.0, 150.0), lams=(0.0, 6.0, 24.0, 48.0), seeds=2)
    d = np.array([v["depth"] for v in G.values()])
    e = np.array([v["e_ctrl"] for v in G.values()])
    assert m.spearman(d, e) > 0.8
    fin = [v["e_ctrl"] for v in G.values() if v["depth"] >= m.DEPTH_BAR]
    unfin = [v["e_ctrl"] for v in G.values() if v["depth"] < m.DEPTH_BAR]
    assert min(fin) > np.median(unfin)               # 완주하는 칸은 전부 비싸다


def test_the_one_knob_habit_hid_a_cheaper_corner():
    """**1 차원으로만 쓸어서 못 본 것** — 같은 깊이에서 더 싼 자리가 있었다."""
    chain = m.clean(seeds=SEEDS)
    corner = m.clean(seeds=SEEDS, d_s=20.0, lam_pos=48.0)
    assert corner["depth"] >= m.DEPTH_BAR
    assert corner["depth"] == pytest.approx(chain["depth"], abs=1.0)   # 같은 깊이인데
    assert corner["e_ctrl"] < chain["e_ctrl"] * 0.8                   # 주입이 20% 넘게 싸다


def test_the_cheaper_corner_is_not_free():
    """싸다고 적었으면 **어느 축에서 무는지**도 같이 고정한다 — exp 63 이후의 규칙이다."""
    chain = m.clean(seeds=SEEDS)
    corner = m.clean(seeds=SEEDS, d_s=20.0, lam_pos=48.0)
    assert corner["osc"] > chain["osc"]
    assert corner["perr"] > chain["perr"]


# --------------------------------------------------------------------------- #
# 5. exp 66 의 관측기는 곡선 밖의 점이다
# --------------------------------------------------------------------------- #
def test_the_observer_reaches_a_cell_the_two_gains_cannot():
    """**exp 66 의 결과를 다시 읽는다.** 관측기가 산 것은 좋은 자리가 아니라 **없는 자리**다."""
    po = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True)
    assert po["e_ctrl"] <= 1e-9 and po["depth"] >= m.DEPTH_BAR
    G = m.scan(ds=(10.0, 30.0, 60.0, 150.0), lams=(0.0, 6.0, 24.0, 48.0), seeds=2)
    assert not any(v["e_ctrl"] <= 1e-9 and v["depth"] >= m.DEPTH_BAR for v in G.values())


def test_the_observer_also_pays_in_position_error():
    """관측기의 대가는 exp 66 이 적은 떨림만이 아니다 — 위치 추종도 문다."""
    chain = m.clean(seeds=SEEDS)
    po = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True)
    assert po["perr"] > chain["perr"]


# --------------------------------------------------------------------------- #
# 6. 남은 2/12 는 이득으로 안 닫힌다 (예측 2 가 틀렸다)
# --------------------------------------------------------------------------- #
def test_the_remaining_violation_is_the_same_seeds_at_every_operating_point():
    """**예측 2 가 틀린 자리.** 운전점을 옮겨도 같은 시드가 깨진다 — 이득의 문제가 아니다."""
    def bad(**kw):
        rs = m.harsh_seeds(seeds=m.HARM_SEEDS, drift_mode="po", po_strict=True, **kw)
        return {i for i, r in enumerate(rs) if r["e_ctrl_max"] > 1e-9}

    at_chain = bad()
    assert at_chain                                   # 깨지는 시드가 실제로 있고
    assert bad(d_s=20.0, lam_pos=48.0) == at_chain    # 값싼 모서리에서도 같은 시드다


# --------------------------------------------------------------------------- #
# 7. 통계 도구 자체의 대조
# --------------------------------------------------------------------------- #
def test_the_rank_correlation_is_a_rank_correlation():
    """직접 짠 순위상관이 맞는지 — 단조 변환에 불변이고 부호가 뒤집혀야 한다."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m.spearman(x, x ** 3) == pytest.approx(1.0)
    assert m.spearman(x, -x) == pytest.approx(-1.0)


def test_main_runs():
    """quick 모드로 전 절이 끝까지 도는지 — 표·그림까지 포함한 연기 시험."""
    out = m.main(quick=True)
    assert set(out) == {"A", "B", "C", "G", "E", "F", "rho"}
    assert Path("assets/67_the_leak_is_the_task.png").exists()
