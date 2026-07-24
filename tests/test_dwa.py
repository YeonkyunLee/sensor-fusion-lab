"""DWA 동적 장애물 회피 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("dwa20", ROOT / "scripts" / "20_dwa_dynamic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dwa_reaches_goal_without_collision():
    mod = _load()
    reached, min_clearance = mod.main()
    assert reached                 # 움직이는 장애물 사이로 목표 도달
    assert min_clearance > 0       # 충돌 없음(항상 양의 여유거리 유지)
