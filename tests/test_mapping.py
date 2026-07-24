"""점유 격자 지도(occupancy-grid mapping, scan-to-map) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "mapping25", ROOT / "scripts" / "25_occupancy_mapping.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_occupancy_map_quality():
    m = _load()
    iou_true, iou_naive, iou_ref = m.main(plot=False)

    # 참 pose로 세운 지도는 벽·기둥을 뚜렷이 복원(관대한 1셀 허용 IoU > 0.5)
    assert iou_true > 0.5
    # scan-to-map 보정이 잡음 pose naive 지도를 개선
    assert iou_ref > iou_naive
    # 잡음 pose 지도는 번져서 참 pose 지도보다 확연히 나쁨(고칠 여지가 있음)
    assert iou_naive < iou_true
