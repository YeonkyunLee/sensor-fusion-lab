"""사전지식을 관측으로 검증하기(exp 52) 테스트. 실 MR 데이터가 없고 못 받으면 skip.

실행: pytest -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "probe52", ROOT / "scripts" / "52_probing_the_prior.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 데이터 없이 도는 코어 테스트
# --------------------------------------------------------------------------- #
def test_deep_mode_is_invisible_at_the_surface_by_construction():
    """실험의 전제: 심부 모드가 표면에 남기는 자취는 프로브 잡음보다 작아야 한다.

    이게 깨지면 '표면으로는 원리적으로 알 수 없다'는 주장 자체가 성립하지 않는다."""
    m = _load()
    window_c = np.zeros(3)
    inward = np.array([0.0, 0.0, -1.0])
    sag = np.array([0.0, 0.0, -1.0])
    _, lat, _ = m.axis_frame(inward)

    surf = np.zeros((200, 3))                     # 창 주변 표면(깊이 0 평면)
    rng = np.random.default_rng(0)
    surf[:, 0] = rng.uniform(-0.04, 0.04, 200)
    surf[:, 1] = rng.uniform(-0.04, 0.04, 200)
    trace = np.linalg.norm(m.deformation(surf, window_c, sag, inward, lat, m.DEEP_MM)
                           - m.deformation(surf, window_c, sag, inward, lat, 0.0), axis=1)
    assert trace.max() < m.PROBE_NOISE, f"표면 자취 {trace.max()*1e3:.2f} mm 가 너무 크다"

    deep = window_c + m.DEEP_DEPTH_MM * 1e-3 * inward
    at_depth = np.linalg.norm(m.deformation(deep[None], window_c, sag, inward, lat, m.DEEP_MM)
                              - m.deformation(deep[None], window_c, sag, inward, lat, 0.0))
    assert at_depth > 20 * trace.max(), "심부/표면 대비가 충분하지 않다"


def test_deep_mode_is_orthogonal_to_the_surface_shift():
    """심부 모드는 표면 침하와 **다른 방향**이다 — 그래서 매끄러운 외삽으로 못 만든다."""
    m = _load()
    inward = np.array([0.0, 0.0, -1.0])
    a, lat, lat2 = m.axis_frame(inward)
    for u, v in ((a, lat), (a, lat2), (lat, lat2)):
        assert abs(float(u @ v)) < 1e-12
    assert abs(np.linalg.norm(lat) - 1.0) < 1e-12


def test_cone_points_respect_the_ultrasound_geometry():
    """초음파가 볼 수 있는 영역: 깊이 구간 안, 원뿔 각 안."""
    m = _load()
    window_c = np.array([0.01, -0.02, 0.05])
    inward = np.array([0.3, -0.2, -1.0]); inward /= np.linalg.norm(inward)
    pts = m.cone_points(np.random.default_rng(3), 500, window_c, inward)
    rel = pts - window_c
    depth = rel @ inward
    lo, hi = m.US_DEPTH_RANGE
    assert depth.min() >= lo - 1e-9 and depth.max() <= hi + 1e-9
    lateral = np.linalg.norm(rel - depth[:, None] * inward[None, :], axis=1)
    assert np.degrees(np.arctan2(lateral, depth)).max() <= m.US_CONE_DEG + 1e-6


# --------------------------------------------------------------------------- #
# 실 MR 데이터 기반
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def result():
    m = _load()
    nrrd = ROOT / m.anat.MRHEAD
    if not nrrd.exists():
        try:
            m.anat.fetch(dest=nrrd)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"공개 MR 데이터 없음(오프라인): {e}")
    return m, m.main(quick=True, n_trials=48)


def test_surface_only_cannot_recover_the_deep_mode(result):
    """핵심 1: 사전지식이 있든 없든 표면만으로는 심부 모드를 복원하지 못한다."""
    _, res = result
    bad = res["sweep"]["심부 모드"]
    assert bad["prior"][0] > 3 * res["surf"]["표면설명형"][1], \
        f"심부 모드가 충분히 아프지 않다: {bad['prior'][0]*1e3:.2f} mm"
    assert abs(bad["prior"][0] - bad["data"][0]) < 0.5e-3, \
        "관측 0개에서 두 방법이 달라지면 안 된다(둘 다 표면만 본다)"


def test_the_prior_still_pays_when_the_surface_explains_the_shift(result):
    """exp 51 의 결과가 여기서도 성립: 표면설명형에서는 사전지식이 이긴다."""
    _, res = result
    ok = res["sweep"]["표면설명형"]
    assert ok["prior"][0] < ok["data"][0], \
        f"사전지식 이득 없음: {ok['prior'][0]*1e3:.2f} vs {ok['data'][0]*1e3:.2f} mm"


def test_depth_observations_are_the_only_cure(result):
    """관측을 넣으면 심부 모드가 실제로 내려온다(사전지식은 못 하는 일)."""
    _, res = result
    bad = res["sweep"]["심부 모드"]
    assert bad["both"][-1] < 0.6 * bad["both"][0], \
        f"관측을 넣어도 안 내려간다: {bad['both'][0]*1e3:.2f} → {bad['both'][-1]*1e3:.2f} mm"
    assert max(bad["prior"]) - min(bad["prior"]) < 1e-3, \
        "사전지식만 쓰는 곡선은 관측 수와 무관해야 한다"


def test_the_surface_residual_is_blind_to_the_violation(result):
    """핵심 2: 볼 수 있는 지표(표면 잔차)는 두 환자에서 사실상 같다."""
    _, res = result
    s_ok, t_ok = res["surf"]["표면설명형"]
    s_bad, t_bad = res["surf"]["심부 모드"]
    assert abs(s_bad / s_ok - 1.0) < 0.05, f"표면 잔차가 {s_bad/s_ok:.2f}배로 벌어졌다"
    assert t_bad / t_ok > 3.0, f"심부 오차는 벌어져야 한다: {t_bad/t_ok:.1f}배"


def test_a_depth_check_beats_the_surface_check(result):
    """핵심 3: 심부 검산 관측이 표면 지표보다 훨씬 잘 잡는다.

    (AUROC 로 재는 이유: 고정 오경보에서의 검출률은 표본 수십 명 규모에서 분위수 추정
     잡음이 커 순위가 뒤집혔다 — 실제로 처음 구현이 그랬다.)"""
    _, res = result
    assert res["gates"][1]["auroc"] > 0.70, \
        f"심부 검산 판별력 부족: AUROC {res['gates'][1]['auroc']:.2f}"
    assert res["gates"][1]["auroc"] > res["auroc_surface"] + 0.10, \
        f"표면 게이트(AUROC {res['auroc_surface']:.2f}) 대비 이득 부족"


def test_more_check_points_do_not_hurt(result):
    """RMS 통계량이면 검산점을 늘릴수록 판별력이 나빠지지 않는다.

    (최댓값 통계량을 쓰던 초기 구현은 여기서 비단조였다 — 잡음이 최댓값에 모인다.)"""
    _, res = result
    g = res["gates"]
    assert g[3]["auroc"] >= g[1]["auroc"] - 0.02, \
        f"점을 늘렸는데 판별이 떨어진다: {g[1]['auroc']:.2f} → {g[3]['auroc']:.2f}"


def test_where_you_check_matters(result):
    """exp 49 의 교훈이 심부에서 반복: 표적 근처 검산점이 아무 데나보다 낫다."""
    _, res = result
    assert res["auroc_target1"] >= res["auroc_random1"], \
        f"표적 근처 {res['auroc_target1']:.2f} vs 무작위 {res['auroc_random1']:.2f}"
    for k in ("auroc_target1", "auroc_random1"):     # 둘 다 표면 게이트보다는 훨씬 낫다
        assert res[k] > res["auroc_surface"] + 0.10, f"{k}={res[k]:.2f}"


def test_knowing_is_cheaper_than_fixing_at_the_first_observation(result):
    """관측 하나는 정확도보다 **지식**을 더 많이 산다 — 판별은 뛰지만 오차는 여전히 허용치 밖."""
    _, res = result
    assert res["gates"][1]["auroc"] - res["auroc_surface"] > 0.10
    assert res["median_fix1_deep"] > res["MISS_TOL"], \
        f"관측 1개로 이미 허용치 안({res['median_fix1_deep']*1e3:.2f} mm)이면 전제가 깨진다"
