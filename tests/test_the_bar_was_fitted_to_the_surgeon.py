"""exp 69 의 주장을 회귀로 고정한다.

주장은 넷이다.
  1. **이 사슬은 표적에 도달한 적이 없다** — 선언된 허용 오차로 채점하면 전부 실패한다.
  2. **시간 탓도 제어기 탓도 아니다** — 정상상태이고, 이득에 반비례하지 않는다.
  3. **상한을 정하는 것은 술자다** — 부족분이 |f_m|/K_OP 와 맞고, 그 항을 쓸면 창에 들어온다.
  4. **바닥은 방향을 구분하지 못한다** — 미달과 초과가 같은 바를 나란히 통과한다.

그리고 이 실험은 **대조가 대조인지부터** 고정한다. 처음에 술자 강성을 상위 모듈에서 패치했는데
시뮬레이터가 import 시점 복사본을 쓰고 있어서 다섯 배율이 전부 같은 숫자를 냈고, 그대로였으면
"술자 강성은 무관하다"는 정반대 결론이 나올 뻔했다. 그래서 인자가 **실제로 먹는지**가 첫 테스트다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("69_the_bar_was_fitted_to_the_surgeon")
jc = import_module("56_jittery_channel")

SEEDS = 3


# --------------------------------------------------------------------------- #
# 0. 공용 시뮬레이터를 또 건드렸으니 — 그리고 대조가 대조인지부터
# --------------------------------------------------------------------------- #
def test_the_new_knob_did_not_change_the_earlier_numbers():
    """k_op 인자를 넣고도 exp 65~68 이 보고한 실행이 숫자까지 같아야 한다."""
    r = jc.run("tdpa", seed=0, jitter_ms=20.0)
    assert r["final_depth_mm"] == pytest.approx(50.76, abs=0.02)
    assert r["e_ctrl_max"] * 1e3 == pytest.approx(80.57, abs=0.05)


def test_the_operator_stiffness_argument_actually_binds():
    """**대조가 대조인지 확인한다.**

    이 실험을 시작할 때 상위 모듈의 상수를 패치했는데 시뮬레이터는 import 시점 복사본을 쓰고
    있어서 **배율을 16 배까지 올려도 결과가 비트 단위로 같았다.** 그대로 읽었으면 "술자 강성은
    무관하다"는 정반대 결론이 나온다. 인자가 먹지 않는 스윕은 스윕이 아니다.
    """
    a = jc.run("tdpa", seed=0, jitter_ms=20.0, steps=8000)
    b = jc.run("tdpa", seed=0, jitter_ms=20.0, steps=8000, k_op=jc.K_OP * 8)
    assert b["final_depth_mm"] > a["final_depth_mm"] + 1.0     # 실제로 달라지고
    c = jc.run("tdpa", seed=0, jitter_ms=20.0, steps=8000, k_op=jc.K_OP)
    assert c["final_depth_mm"] == a["final_depth_mm"]          # 기본값은 그대로다


# --------------------------------------------------------------------------- #
# 1. 창으로 채점하면 전부 실패한다
# --------------------------------------------------------------------------- #
def test_the_window_half_width_comes_from_a_declared_constant():
    """**지어낸 숫자가 아니다** — exp 45 가 선언한 MISS_TOL 을 그대로 쓴다(exp 58 의 규칙)."""
    g6 = import_module("45_image_guided_6dof")
    assert m.WINDOW_MM == pytest.approx(g6.MISS_TOL * 1e3)
    assert m.DEPTH_BAR < m.TARGET_MM - m.WINDOW_MM     # 바닥이 창보다 한참 아래에 있다


def test_the_chain_baseline_passes_the_floor_and_fails_the_window():
    """**이 실험의 결과.** 열두 실험이 완주라 부른 운전점이 표적 허용 오차 밖이다."""
    base = m.depth(seeds=SEEDS)
    assert base["floor_pass"]
    assert not base["window_pass"]
    assert base["err"] < -m.WINDOW_MM                  # 미달 쪽으로 벗어난다


def test_no_configuration_the_chain_used_lands_in_the_window():
    """사슬이 실제로 써 온 세 운전점 중 창에 드는 것이 없다."""
    for _, kw in m.CONFIGS[:4]:
        assert not m.depth(seeds=SEEDS, **kw)["window_pass"]


def test_a_final_value_metric_cannot_see_the_excursion():
    """**두 번째 결함.** exp 68 의 칸은 표적을 크게 지나갔다가 **돌아온다.**

    그래서 정상상태까지 돌리면 최종 깊이가 기준선과 사실상 같아지고, 사슬의 지표로는 두 실행이
    구분되지 않는다. exp 68 은 4 초 실행의 최종값이 우연히 높아서 초과를 봤다 — 이유가 달랐다.
    조직에 무엇을 했는가는 **어디서 끝났는가**가 아니라 **어디까지 갔는가**의 문제다.
    """
    base = m.depth(seeds=SEEDS)
    cell = m.depth(seeds=SEEDS, **dict(m.CONFIGS[3][1]))
    assert cell["depth"] == pytest.approx(base["depth"], abs=1.0)    # 최종값은 구분이 안 되는데
    assert cell["peak"] > base["peak"] + 5.0                         # 경로는 전혀 다르다
    assert cell["peak"] > m.TARGET_MM + m.WINDOW_MM                  # 표적을 창 밖으로 지나갔다
    assert not cell["peak_pass"] and base["peak_pass"]


def test_the_floor_passes_both_directions():
    """**바닥의 결함** — 미달한 실행과 표적을 지나간 실행이 같은 바를 나란히 통과한다."""
    base = m.depth(seeds=SEEDS)
    cell = m.depth(seeds=SEEDS, **dict(m.CONFIGS[3][1]))
    assert base["floor_pass"] and cell["floor_pass"]                 # 바닥은 둘 다 통과시키고
    assert not base["window_pass"] and not cell["peak_pass"]         # 창은 둘을 다른 이유로 잡는다


# --------------------------------------------------------------------------- #
# 2. 시간도 제어기도 아니다
# --------------------------------------------------------------------------- #
def test_the_shortfall_is_a_steady_state_not_a_time_limit():
    """**실패할 수 있는 시험으로 만들기 위한 대조**(R18 의 원래 취지)."""
    short = m.depth(seeds=SEEDS, steps=8000)
    long = m.depth(seeds=SEEDS, steps=32000)
    assert long["depth"] == pytest.approx(short["depth"], abs=0.05)   # 8 배 시간에도 같고
    assert abs(long["creep"]) < 1e-3                                  # 도구가 멈춰 있다


def test_the_error_saturates_instead_of_scaling_with_the_gains():
    """**예측이 틀린 자리.** 표류 오차라면 1/(d_s·λ) 로 줄어야 하는데 포화한다."""
    lo = m.depth(seeds=SEEDS, d_s=60.0, lam_pos=24.0)
    hi = m.depth(seeds=SEEDS, d_s=240.0, lam_pos=384.0)               # 이득 640 배
    assert abs(hi["err"]) > m.WINDOW_MM                               # 여전히 창 밖이고
    assert abs(hi["err"]) > 0.7 * abs(lo["err"])                      # 거의 안 준다


# --------------------------------------------------------------------------- #
# 3. 상한은 술자가 정한다
# --------------------------------------------------------------------------- #
def test_the_master_shortfall_matches_the_hand_stiffness_prediction():
    """정상상태 부족분이 |f_m|/K_OP 와 맞아야 귀속이라 부를 수 있다."""
    base = m.depth(seeds=SEEDS)
    predicted = abs(base["f_m"]) / jc.K_OP * 1e3
    assert (m.TARGET_MM - base["master"]) == pytest.approx(predicted, abs=0.05)


def test_most_of_the_shortfall_is_the_operator_not_the_controller():
    """부족분의 대부분이 마스터 쪽이다 — 제어기를 고쳐서는 못 닫는다는 근거."""
    base = m.depth(seeds=SEEDS)
    master_share = (m.TARGET_MM - base["master"]) / (m.TARGET_MM - base["depth"])
    assert master_share > 0.7


def test_sweeping_the_operator_stiffness_reaches_the_window():
    """귀속의 확인 — 그 항을 쓸면 실제로 창에 들어온다(처방이라는 뜻은 아니다)."""
    assert not m.depth(seeds=SEEDS, k_op=jc.K_OP)["window_pass"]
    assert m.depth(seeds=SEEDS, k_op=jc.K_OP * 8)["window_pass"]


def test_the_transmitted_force_is_what_pushes_the_master_back():
    """**사슬의 성과가 사슬의 한계다** — 표시 힘이 없으면 마스터가 밀리지 않는다."""
    base = m.depth(seeds=SEEDS)
    assert abs(base["f_m"]) > 0.5                       # 실제로 힘이 전달되고 있고
    assert base["master"] < m.TARGET_MM - 1.0           # 그만큼 마스터가 뒤에 선다


def test_main_runs():
    """quick 모드로 전 절이 끝까지 도는지 — 표·그림까지 포함한 연기 시험."""
    out = m.main(quick=True)
    assert set(out) == {"A", "B", "C", "E", "window", "target"}
    assert Path("assets/69_the_bar_was_fitted_to_the_surgeon.png").exists()
