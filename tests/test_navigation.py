"""계획+제어 내비게이션 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_plan_control_reaches_goal_avoids_nogo():
    spec = importlib.util.spec_from_file_location("nav19", ROOT / "scripts" / "19_plan_control.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    reached, clearance = mod.main()
    assert reached                # 목표 도달
    assert clearance > 0          # no-go 존 침범 없음


def test_astar_returns_none_when_blocked():
    spec = importlib.util.spec_from_file_location("nav19b", ROOT / "scripts" / "19_plan_control.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy as np
    blocked = np.ones((20, 20), bool)   # 전부 장애물
    assert mod.astar(blocked, (1, 1), (8, 8)) is None
