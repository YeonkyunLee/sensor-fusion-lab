"""전체 2D LiDAR SLAM(ICP 프론트엔드 + pose-graph 백엔드) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "slam23", ROOT / "scripts" / "23_lidar_slam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backend_reduces_frontend_drift():
    m = _load()
    odom_rmse, opt_rmse, n_lc, chi2_before, chi2_after = m.main(plot=False)

    # 프론트엔드가 눈에 띄게 드리프트(백엔드로 고칠 여지가 있어야 함)
    assert odom_rmse > 0.4
    # 장소재인식이 실제 루프클로저를 여럿 찾음
    assert n_lc >= 5
    # 백엔드가 프론트엔드 드리프트를 뚜렷이 줄임(절반 미만) + 절대 기준 통과
    assert opt_rmse < 0.5 * odom_rmse
    assert opt_rmse < 0.4
    # 최적화가 제약 오차(chi2)를 크게 감소
    assert chi2_after < chi2_before
