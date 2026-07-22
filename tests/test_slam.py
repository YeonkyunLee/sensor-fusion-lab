"""EKF-SLAM 통합 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_slam():
    spec = importlib.util.spec_from_file_location("slam05", ROOT / "scripts" / "05_ekf_slam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_slam_beats_odometry_and_maps():
    slam = _load_slam()
    traj_rmse, odo_rmse, map_err = slam.main(plot=False)
    # SLAM이 순수 오도메트리보다 뚜렷이 정확
    assert traj_rmse < odo_rmse * 0.5
    # 지도(랜드마크) 오차도 작아야 함
    assert map_err < 1.0


def _load_loop():
    spec = importlib.util.spec_from_file_location("loop06", ROOT / "scripts" / "06_loop_closure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loop_closure_reduces_return_drift():
    loop = _load_loop()
    ret_no, ret_yes = loop.main()
    # 루프 클로저가 복귀 구간 드리프트를 줄여야 함
    assert ret_yes < ret_no
