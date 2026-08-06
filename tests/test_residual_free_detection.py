"""exp 64 의 주장과 **정정**을 회귀로 고정한다.

이 실험의 결론은 네거티브다 — exp 55 가 제안한 대체 검출기(부분집합 일치)가 **이미 있던 잔차보다
낫지 않다.** 그리고 exp 55 의 *"흔적을 전혀 안 남긴다"* 는 문구가 과했다는 정정이 붙는다.
설계 중 밟은 함정(검출 대상 하나만 흔들어 AUROC 1.00)도 같이 박아 둔다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
m = import_module("64_residual_free_detection")
s55 = import_module("55_correspondence_search")

CTX = m.build(seed=5)


def _draw(n, tan_hi=8.0, nrm_lo=2.0, nrm_hi=10.0, oracle=False):
    """법선·접선을 **독립으로** 뽑아 시행한다 — 그래야 검출 과제가 성립한다."""
    out = []
    for i in range(n):
        rr = np.random.default_rng([5, 4000 + i])
        nrm = float(rr.uniform(nrm_lo, nrm_hi))
        tan = float(rr.uniform(0.0, tan_hi))
        out.append((nrm, tan) + m.trial(CTX, rr, tan, oracle=oracle, normal_mm=nrm))
    return np.array(out, float)


# --------------------------------------------------------------------------- #
# 도구가 제대로 서 있는가
# --------------------------------------------------------------------------- #
def test_sectors_partition_the_observations():
    """부채꼴은 겹치지 않고 전부 덮어야 한다 — 안 그러면 '독립 부분집합'이 아니다."""
    rr = np.random.default_rng(1)
    _, src, _ = m.observe(CTX, rr, 4.0)
    parts = m.sectors(CTX, src)
    all_idx = np.concatenate(parts)
    assert len(all_idx) == len(src)
    assert len(np.unique(all_idx)) == len(src)
    assert all(len(p) > 20 for p in parts), [len(p) for p in parts]


def test_auroc_is_correct_on_a_known_case():
    assert m.auroc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert m.auroc([4, 3, 2, 1], [0, 0, 1, 1]) == pytest.approx(0.0)
    # 양성이 음성 안쪽에 끼면 정확히 우연 수준이다(양성 2·3 vs 음성 1·4 → 2/4).
    assert m.auroc([1, 2, 3, 4], [0, 1, 1, 0]) == pytest.approx(0.5)


def test_setup_matches_exp55():
    """exp 55 와 같은 표면·노출·감쇠를 쓰는지 — 아니면 그 실험의 결론을 시험하는 게 아니다."""
    assert m.NORMAL_MM == s55.NORMAL_MM
    assert m.DECAY_MM == s55.DECAY_MM
    assert m.EXPOSURE_DEG == s55.EXPOSURE_DEG
    assert m.TARGET_DEPTHS_MM == s55.TARGET_DEPTHS_MM


# --------------------------------------------------------------------------- #
# exp 55 의 전제와 그 정정
# --------------------------------------------------------------------------- #
def test_target_error_grows_much_faster_than_the_residual():
    """전제 재확인: 접선을 키우면 표적 오차가 잔차보다 **훨씬 빨리** 는다."""
    lo = [m.trial(CTX, np.random.default_rng([5, 10, r]), 0.0) for r in range(3)]
    hi = [m.trial(CTX, np.random.default_rng([5, 80, r]), 8.0) for r in range(3)]
    e_lo, e_hi = np.median([x[0] for x in lo]), np.median([x[0] for x in hi])
    s_lo, s_hi = np.median([x[1] for x in lo]), np.median([x[1] for x in hi])
    assert e_hi > e_lo * 2.5                       # 오차는 배로 늘고
    assert s_hi < s_lo * 1.6                       # 잔차는 조금만 는다


def test_the_residual_is_insensitive_not_blind():
    """**exp 55 문구의 정정.** 잔차가 접선에 대해 0 이 아니라 단조로 오른다."""
    vals = [np.median([m.trial(CTX, np.random.default_rng([5, int(t * 10), r]), t)[1]
                       for r in range(3)]) for t in (0.0, 4.0, 8.0)]
    assert vals[-1] > vals[0] * 1.10               # 확실히 오르고
    assert vals[-1] < vals[0] * 1.60               # 그러나 오차만큼은 아니다


# --------------------------------------------------------------------------- #
# 설계 함정 — 실패할 수 없는 검출 과제
# --------------------------------------------------------------------------- #
def test_varying_only_the_target_makes_the_task_unfailable():
    """**밟은 함정.** 접선만 흔들면 잔차가 만점에 가까워진다 — 시험이 실패할 수가 없다.

    exp 56 이 채널에서 발견한 것과 같은 계열. 그래서 법선도 함께 흔들어야 과제가 성립한다.
    """
    rows = np.array([(t,) + m.trial(CTX, np.random.default_rng([5, int(t * 10), r]), t)
                     for t in (0.0, 4.0, 8.0) for r in range(4)], float)
    lab = rows[:, 1] > np.median(rows[:, 1])
    easy = m.auroc(rows[:, 2], lab)
    hard = m.auroc(_draw(40)[:, 3], _draw(40)[:, 2] > np.median(_draw(40)[:, 2]))
    assert easy > 0.95                             # 접선만 흔들면 거의 만점이고
    assert hard < easy                             # 법선을 섞으면 내려온다


# --------------------------------------------------------------------------- #
# 결론 — 네거티브
# --------------------------------------------------------------------------- #
def test_subset_agreement_does_not_beat_the_residual():
    """**이 실험의 결론.** 제안된 대체 검출기가 이미 있던 것보다 낫지 않다."""
    d = _draw(40)
    lab = d[:, 2] > np.median(d[:, 2])
    a_res = m.auroc(d[:, 3], lab)
    a_spr = m.auroc(d[:, 4], lab)
    assert a_res > 0.7, f"잔차가 쓸 만해야 정정이 성립한다: {a_res:.2f}"
    assert a_spr < a_res, f"불일치 {a_spr:.2f} vs 잔차 {a_res:.2f}"


def test_the_two_statistics_are_not_independent_axes():
    """exp 63 의 기준 — 다른 축이라면 순위 상관이 낮아야 하는데 그렇지 않다."""
    d = _draw(40)
    assert m._rho(d[:, 3], d[:, 4]) > 0.25


def test_the_control_shows_a_floor_of_ill_conditioning():
    """**exp 61 의 R26.** 정답 대응에서도 불일치가 남는다 — 절대값 문턱은 조건수를 잰다."""
    orc = _draw(12, tan_hi=0.01, oracle=True)
    floor = float(np.nanmedian(orc[:, 4]))
    assert floor > 0.2e-3, "대조군이 0 이면 병조건 우려가 없었다는 뜻"
    nn = _draw(12, tan_hi=0.01, oracle=False)
    assert float(np.nanmedian(nn[:, 4])) < floor * 3.0


def test_oracle_correspondence_lowers_the_target_error():
    """정답 대응이 실제로 더 나은지 — 아니면 이 실험의 대조군이 대조군이 아니다."""
    nn = _draw(10, oracle=False)
    orc = _draw(10, oracle=True)
    assert np.nanmedian(orc[:, 2]) < np.nanmedian(nn[:, 2])
