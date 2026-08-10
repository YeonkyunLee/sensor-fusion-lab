"""exp 66 의 주장을 회귀로 고정한다.

주장은 셋이다.
  1. **아홉 실험이 지목한 항을 죄어서는 보증이 오지 않는다** — 어떤 탱크 용량에서도.
  2. **두 포트 장부를 직접 감시하면 온다** — 과제를 완주한 채로.
  3. **고치는 것도 자기가 보는 장부만 고친다** — exp 65 의 교훈이 처방 쪽에서 반복된다.

그리고 이 실험은 공용 시뮬레이터(exp 56 의 run)를 건드렸으므로, **앞 실험들의 숫자가 그대로인지**
부터 고정한다 — 실제로 리팩터 중에 반사파 송신 한 줄을 떨어뜨려 깊이가 50.8 → 36.7 mm 로 바뀐
적이 있고, 그때 이 파일의 첫 테스트가 그걸 잡는 자리였다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("66_passive_drift_correction")
jc = import_module("56_jittery_channel")

SEEDS = 4


# --------------------------------------------------------------------------- #
# 0. 공용 시뮬레이터를 건드렸으니 앞 실험이 그대로인지부터
# --------------------------------------------------------------------------- #
def test_the_refactor_did_not_change_the_earlier_numbers():
    """exp 65 가 보고한 바로 그 실행이 숫자까지 그대로여야 한다.

    항별 분해를 넣느라 결합력을 파동 몫·표류 몫으로 쪼갰다. 대수적으로 같은 식이지만
    **같은 식이라고 믿는 것과 같은 숫자인 것은 다르다.**
    """
    r = jc.run("tdpa", seed=0, jitter_ms=20.0)
    assert r["final_depth_mm"] == pytest.approx(50.76, abs=0.02)
    assert r["e_ctrl_max"] * 1e3 == pytest.approx(80.57, abs=0.05)
    assert r["e_min"] >= -1e-9                      # 파동 블록은 여전히 만족


def test_the_decomposition_is_a_decomposition():
    """항별 합이 전체 장부와 **정확히** 같아야 한다 — 아니면 분해가 아니라 추정이다."""
    for kw in ({}, dict(drift_mode="po"), dict(drift_mode="tank")):
        r = jc.run("tdpa", seed=1, jitter_ms=20.0, **kw)
        assert sum(r["e_term"].values()) == pytest.approx(r["e_ctrl_end"], abs=1e-12)


def test_the_decomposition_holds_when_the_stop_is_blending():
    """exp 58 의 정지가 결합력을 혼합할 때도 합이 유지되는지 — 혼합을 항별로 안 걸면 깨진다."""
    bc = import_module("57_bursty_channel")
    s59 = import_module("59_what_is_safe_state")
    r = bc.run("tdpa", seed=0, tissue_obj=s59.GrippingTissue(), estop=True,
               tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS,
               resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=s59.BREATH_HZ)
    assert r["n_estop"] > 0                          # 정지가 실제로 걸린 실행이어야 의미가 있다
    assert sum(r["e_term"].values()) == pytest.approx(r["e_ctrl_end"], abs=1e-12)


# --------------------------------------------------------------------------- #
# 1. 지목된 항을 죄어서는 오지 않는다
# --------------------------------------------------------------------------- #
def test_the_baseline_really_is_active():
    """**실패할 수 없는 시험을 만들지 않기 위한 대조.** 기준선이 능동이어야 아래가 의미 있다."""
    assert m.clean(seeds=SEEDS)["e_ctrl"] > 0.02


@pytest.mark.parametrize("cap", [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1])
def test_the_tank_never_reaches_passivity_at_any_capacity(cap):
    """**핵심 부정 결과.** 정직하게 재원을 댄 탱크도 어느 용량에서 수동이 되지 않는다."""
    assert m.clean(seeds=SEEDS, drift_mode="tank", tank_max=cap)["e_ctrl"] > 1e-9


def test_turning_the_named_term_fully_off_still_leaves_injection():
    """용량 0 = 표류 항 완전 차단. 그래도 남는다 — **지목이 부분적으로만 옳았다는 증거.**

    exp 65 는 λ 를 쓸어 상관을 봤다. 이건 꺼서 인과를 본 것이고, 답이 다르다.
    """
    off = m.clean(seeds=SEEDS, drift_mode="tank", tank_max=0.0)
    assert off["e_ctrl"] * 1e3 > 10.0               # 10 mJ 넘게 남고
    assert off["depth"] < 40.0                      # 과제는 못 한다 (R18 45 mm 아래)


def test_a_large_tank_never_binds():
    """용량이 크면 제약이 아예 안 걸린다 — '수동성을 얻었다'가 아니라 '아무것도 안 했다'."""
    big = m.clean(seeds=SEEDS, drift_mode="tank", tank_max=1e-1)
    raw = m.clean(seeds=SEEDS)
    assert big["tank_dry"] < 0.02
    assert big["e_ctrl"] == pytest.approx(raw["e_ctrl"], rel=0.05)


def test_partial_gating_loses_on_both_axes():
    """중간 용량이 가장 나쁘다 — 켜졌다 꺼졌다 하는 것 자체가 진동원이다."""
    mid = m.clean(seeds=SEEDS, drift_mode="tank", tank_max=3e-3)
    raw = m.clean(seeds=SEEDS)
    assert mid["osc"] > raw["osc"] * 3.0
    assert mid["e_ctrl"] > raw["e_ctrl"]


# --------------------------------------------------------------------------- #
# 2. 장부를 직접 감시하면 온다
# --------------------------------------------------------------------------- #
def test_the_observer_restores_passivity_on_every_seed_and_every_time():
    """**이 실험의 결과.** 수동성은 모든 실행·모든 시각 조건이라 최악 시드로 묻는다."""
    for s in range(6):
        r = jc.run("tdpa", seed=s, jitter_ms=20.0, drift_mode="po", po_strict=True)
        assert r["e_ctrl_max"] <= 1e-9, f"seed {s}"


def test_the_guarantee_is_only_as_deep_as_the_seeds_that_were_run():
    """**수동성은 본 만큼만 참이다.** 최악 조건에서는 6 시드까지 0 이고 12 시드에서 위반이 나온다.

    exp 65 는 중앙값이 위반의 절반을 숨기는 것을 만났다. 같은 함정이 여기서는 **시드 수**로 온다 —
    이 테스트가 없으면 위 `..._on_every_seed_and_every_time` 이 조건까지 함께 주장하는 것으로
    읽힌다. 그 문장은 **지터만 있는 채널**에 대한 것이다.
    """
    assert m.harsh(seeds=6, drift_mode="po", po_strict=True)["e_ctrl"] <= 1e-9
    worst = m.harsh(seeds=m.HARM_SEEDS, drift_mode="po", po_strict=True)
    assert worst["e_ctrl"] > 1e-9                                   # 12 시드에서는 나오고
    assert worst["e_ctrl"] < m.harsh(seeds=m.HARM_SEEDS)["e_ctrl"] * 0.05   # 그래도 97% 준다


def test_the_observer_keeps_the_task():
    """보증을 사고 과제를 팔면 exp 56 이 λ 를 올린 이유로 되돌아간다 — R18 을 넘겨야 한다."""
    po = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True)
    assert po["depth"] >= m.DEPTH_BAR
    assert po["diverged"] == 0


def test_the_observer_actually_applies_force_it_is_not_bookkeeping():
    """**장부만 예쁘게 만드는 것이 아닌지.** 관측기가 실제로 소산 일을 해야 한다.

    이 확인이 없으면 '장부가 0'은 계산 방식의 성질일 수 있다.
    """
    r = jc.run("tdpa", seed=0, jitter_ms=20.0, drift_mode="po", po_strict=True)
    assert r["pc_duty"] > 0.3                        # 자주 개입하고
    assert r["e_pc"] < -1e-3                         # 그 개입이 실제로 에너지를 뽑는다


def test_the_price_is_chatter():
    """대가가 있다고 적었으면 그 대가도 고정해 둬야 한다 — 진동이 자릿수로 는다."""
    raw = m.clean(seeds=SEEDS)
    po = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True)
    assert po["osc"] > raw["osc"] * 3.0


def test_lowering_lambda_cannot_buy_passivity_while_finishing():
    """λ 사다리에는 '수동이면서 완주하는' 칸이 없다 — 그래서 관측기가 필요했다."""
    finishing = [m.clean(seeds=SEEDS, lam_pos=lam) for lam in (10.0, 12.0, 16.0, 24.0)]
    assert all(v["depth"] >= m.DEPTH_BAR for v in finishing)
    assert min(v["e_ctrl"] for v in finishing) * 1e3 > 50.0


# --------------------------------------------------------------------------- #
# 3. 고치는 것도 자기가 보는 장부만 고친다
# --------------------------------------------------------------------------- #
def test_the_fix_repairs_only_the_ledger_it_watches():
    """**exp 65 의 교훈이 처방 쪽에서 반복된다.**"""
    lenient = m.clean(seeds=SEEDS, drift_mode="po")
    strict = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True)
    assert lenient["e_ctrl"] * 1e3 < 5.0             # 자기가 보는 쪽은 고치고
    assert lenient["e_nd"] * 1e3 > 40.0              # 안 보는 쪽은 남긴다
    assert strict["e_nd"] * 1e3 < 5.0                # 엄격한 쪽을 보게 하면 둘 다 만족


def test_a_stronger_clamp_is_not_better():
    """처방의 **자기 실패 모드** — 한 스텝에 큰 힘을 때리는 것 자체가 여기가 된다."""
    good = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True, pc_fmax=50.0)
    hard = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True, pc_fmax=5000.0)
    weak = m.clean(seeds=SEEDS, drift_mode="po", po_strict=True, pc_fmax=10.0)
    assert hard["e_nd"] > good["e_nd"] * 10.0        # 세면 나빠지고
    assert weak["e_ctrl"] > good["e_ctrl"]           # 약하면 못 뽑는다


# --------------------------------------------------------------------------- #
# 4. 환자 쪽 — 짝지어 비교
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seeds", [3, 6])
def test_the_harm_gain_is_mostly_depth_not_passivity(seeds):
    """**짝지으면 자랑이 사라진다.** CHECKLIST 의 그 항목이 여기서 한 번 더 일한다.

    시드 수를 바꿔도 같은 답이 나와야 결론이라 부를 수 있다 — 아래 취약한 쪽과 대비된다.
    """
    ladder = {lam: m.harsh(seeds=seeds, lam_pos=lam) for lam in (6.0, 10.0, 16.0, 24.0)}
    po = m.harsh(seeds=seeds, drift_mode="po", po_strict=True)
    raw = m.harsh(seeds=seeds)
    assert po["drag"] < raw["drag"] * 0.8            # 안 짝지으면 크게 좋아 보이는데
    matched, _ = m.depth_matched_drag(ladder, po["depth"])
    # 안 짝지으면 −39% 로 보이던 것이, 짝지으면 십몇 % 안으로 들어온다(부호도 일정하지 않다).
    assert po["drag"] == pytest.approx(matched, rel=0.15)


def test_the_held_interval_gain_is_real_but_seed_fragile():
    """정지 구간 끌림은 12 시드에서 개선으로 나오는데 **3 시드에서는 부호가 뒤집힌다.**

    처음에 이걸 '짝지어도 남는 이득'이라고 3 시드로 단언했다가 이 테스트가 반증했다. 그래서
    주장을 그 축을 세운 실험의 시드 수(exp 63 = 12)로 옮기고, **취약하다는 사실 자체를**
    결과에 적었다. 취약성을 고정해 두지 않으면 다음 사람이 3 시드로 재보고 뒤집힌다.
    """
    def gap(seeds):
        ladder = {lam: m.harsh(seeds=seeds, lam_pos=lam)
                  for lam in (6.0, 10.0, 16.0, 24.0)}
        po = m.harsh(seeds=seeds, drift_mode="po", po_strict=True)
        _, matched_held = m.depth_matched_drag(ladder, po["depth"])
        return po["drag_held"] - matched_held

    assert gap(m.HARM_SEEDS) < 0.0                   # 12 시드에서는 개선이고
    assert gap(3) > 0.0                              # 3 시드에서는 반대로 나온다


def test_main_runs():
    """quick 모드로 전 절이 끝까지 도는지 — 표·그림까지 포함한 연기 시험."""
    out = m.main(quick=True)
    assert set(out) == {"A", "B", "C", "L", "F", "E", "ladder"}
    assert Path("assets/66_passive_drift_correction.png").exists()


def test_the_interpolator_is_paired_not_clever():
    """보간이 실제로 깊이축 보간인지 — 사다리 위의 점을 넣으면 그 점이 나와야 한다."""
    ladder = {1: dict(depth=40.0, drag=5.0, drag_held=2.0),
              2: dict(depth=50.0, drag=15.0, drag_held=4.0)}
    assert m.depth_matched_drag(ladder, 45.0) == pytest.approx((10.0, 3.0))
    assert m.depth_matched_drag(ladder, 40.0) == pytest.approx((5.0, 2.0))
