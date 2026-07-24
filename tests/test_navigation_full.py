"""내비게이션 종합(capstone) 테스트: A* 전역 + 장애물 인지 MPC 지역. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "nav28", ROOT / "scripts" / "28_full_navigation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_full_system_reaches_goal_and_avoids_while_plain_collides():
    mod = _load()
    r = mod.main()

    # 1) 전체 시스템(A* + 장애물 인지 MPC)은 목표에 도달한다.
    assert r["reached_goal"] is True, "full system must reach the goal"

    # 2) 전체 시스템은 이동 장애물과 충돌하지 않는다(여유 > 0).
    assert r["full_min_clearance"] > 0.0, \
        f"full system must not collide: {r['full_min_clearance']}"

    # 3) 동일한 A* 경로를 '장애물 무지' 추종기로 따라가면 이동 장애물과 충돌한다(여유 < 0).
    assert r["plain_min_clearance"] < 0.0, \
        f"plain tracker should collide: {r['plain_min_clearance']}"

    # 4) 전체 시스템이 무지 추종기보다 여유거리를 확실히 크게 확보한다.
    assert r["full_min_clearance"] > r["plain_min_clearance"] + 0.5

    # 5) 전체 시스템은 정적 지도(벽/no-go)도 침범하지 않고 액추에이터 한계를 지킨다.
    assert r["static_clearance"] > 0.0, \
        f"full system must not clip static map: {r['static_clearance']}"
    assert r["limits_ok"] is True


def test_astar_blocked_returns_none():
    mod = _load()
    import numpy as np
    blocked = np.ones((20, 20), bool)
    assert mod.astar(blocked, (1, 1), (8, 8)) is None
