"""ICP scan-matching 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("icp21", ROOT / "scripts" / "21_icp_scan_matching.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_icp_recovers_known_transform():
    m = _load()
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-10, 10, (250, 2))
    Ttrue = m.se2(0.6, -0.4, 0.15)          # 알려진 강체변환
    moved = m.transform(cloud, Ttrue)
    # icp(src=cloud, dst=moved) 는 cloud->moved 변환 = Ttrue 를 복원해야 함
    T, err, _ = m.icp(cloud, moved, max_corr_dist=50.0)
    got = m.se2_params(T)
    exp = m.se2_params(Ttrue)
    assert abs(got[0] - exp[0]) < 1e-2
    assert abs(got[1] - exp[1]) < 1e-2
    assert abs(m.wrap(got[2] - exp[2])) < 1e-3
    assert err < 1e-6                        # 완전 정렬(잡음 없음)


def test_icp_recovers_transform_with_noise_and_partial_overlap():
    m = _load()
    rng = np.random.default_rng(3)
    cloud = rng.uniform(-8, 8, (300, 2))
    Ttrue = m.se2(0.5, 0.3, -0.12)
    moved = m.transform(cloud, Ttrue) + rng.normal(0, 0.02, cloud.shape)
    src = cloud[:230]                        # 부분 겹침(70%가량)
    T, _, _ = m.icp(src, moved, init=m.se2(0.4, 0.2, -0.1), max_corr_dist=2.0)
    got, exp = m.se2_params(T), m.se2_params(Ttrue)
    assert np.hypot(got[0] - exp[0], got[1] - exp[1]) < 0.1
    assert abs(m.wrap(got[2] - exp[2])) < 0.03


def test_icp_odometry_beats_raw_dead_reckoning():
    m = _load()
    icp_rmse, raw_rmse = m.main()
    assert icp_rmse < 0.4              # ICP 오도메트리가 궤적을 정밀 복원
    assert icp_rmse < raw_rmse * 0.5  # 무보정 dead-reckoning보다 뚜렷이 정확
