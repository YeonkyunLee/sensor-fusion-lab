"""Monte Carlo Localization(파티클 필터) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_pf():
    spec = importlib.util.spec_from_file_location(
        "pf36", ROOT / "scripts" / "36_particle_filter.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mcl_beats_odometry_and_globally_converges():
    pf = _load_pf()
    odo_rmse, mcl_rmse, conv_step = pf.main(plot=False)

    # (i) MCL 추적이 순수 오도메트리(dead reckoning)보다 뚜렷이 정확
    assert mcl_rmse < odo_rmse * 0.4
    # (ii) 절대 정확도(수렴)도 확보 — 1m 미만
    assert mcl_rmse < 1.0
    # (iii) 전역(kidnapped) 위치추정: 초기 pose 없이도 빠르게 수렴
    assert conv_step < 25


def test_ring_posterior_is_non_gaussian():
    """단일 거리 관측의 사후분포가 고리형(비가우시안)임을 검증.
    파티클은 반지름 r0 근처에 모이고, 가우시안 평균이 놓이는 중심은 비어 있다."""
    pf = _load_pf()
    lm, r0, P, gmean, gcov = pf.ring_demo(seed=0)
    d = ((P[:, 0] - lm[0]) ** 2 + (P[:, 1] - lm[1]) ** 2) ** 0.5
    # 대부분의 파티클이 관측 거리 근처(고리 위)에 존재
    assert abs(d.mean() - r0) < 2.0
    assert d.std() < 2.0
    # 가우시안 평균(EKF 믿음의 중심)은 고리 중심 = 실제 확률이 없는 '빈 곳'
    import numpy as np
    assert np.hypot(gmean[0] - lm[0], gmean[1] - lm[1]) < 3.0
