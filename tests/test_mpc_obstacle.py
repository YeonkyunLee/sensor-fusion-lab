"""장애물 회피 MPC 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "mpc26", ROOT / "scripts" / "26_mpc_obstacle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_obstacle_aware_avoids_while_plain_collides():
    mod = _load()
    r = mod.main()

    # 1) 무지 MPC(exp24식)는 경로 위 장애물을 관통 -> 충돌(여유 < 0)
    assert r["plain_min_clearance"] < 0.0, \
        f"plain MPC should collide: {r['plain_min_clearance']}"

    # 2) 인지 MPC는 충돌 없음(여유 > 0), 이상적으로 안전여유의 절반 이상 확보
    assert r["collision_avoided"] is True
    assert r["aware_min_clearance"] > 0.0, \
        f"obstacle-aware MPC must not collide: {r['aware_min_clearance']}"
    assert r["aware_min_clearance"] >= 0.5 * mod.MARGIN, \
        f"aware clearance below half-margin: {r['aware_min_clearance']}"

    # 3) 인지 MPC가 무지 MPC보다 여유거리 확실히 큼
    assert r["aware_min_clearance"] > r["plain_min_clearance"] + 0.5


def test_obstacle_aware_still_tracks_and_respects_limits():
    mod = _load()
    Xref, Uref = mod.make_reference()
    nref = len(Xref)
    i1, i2, i3 = int(0.17 * nref), int(0.50 * nref), int(0.80 * nref)
    velC = np.array([-0.35, 0.20])
    pC0 = Xref[i3, :2] - velC * (i3 * mod.DT)
    obs = [
        mod.Obstacle(Xref[i1, :2], [0.0, 0.0], 0.55),
        mod.Obstacle(Xref[i2, :2], [0.0, 0.0], 0.55),
        mod.Obstacle(pC0, velC, 0.55),
    ]
    traj, u = mod.run_mpc(Xref, Uref, obs, beta=mod.BETA, margin=mod.MARGIN)

    # 액추에이터 절대/가속 한계 준수
    v = np.abs(u[:, 0])
    w = np.abs(u[:, 1])
    assert v.max() <= mod.V_MAX + 1e-6, f"|v| exceeds limit: {v.max()}"
    assert w.max() <= mod.W_MAX + 1e-6, f"|w| exceeds limit: {w.max()}"
    assert np.abs(np.diff(u[:, 0])).max() / mod.DT <= mod.A_V + 1e-6
    assert np.abs(np.diff(u[:, 1])).max() / mod.DT <= mod.A_W + 1e-6

    # 장애물에서 먼 구간에선 기준궤적을 조밀하게 추종(회피 후 복귀 확인)
    clear = mod.clearance_series(traj, obs)
    ct = mod.cross_track(traj, Xref)
    far = clear > 1.2
    rmse_far = float(np.sqrt(np.mean(ct[far] ** 2)))
    assert rmse_far < 0.25, f"tracking away from obstacles too loose: {rmse_far}"

    # 진행/복귀: 궤적 끝이 출발점 부근(8자 폐곡선 한 바퀴)으로 돌아옴
    assert np.hypot(*(traj[-1] - Xref[0, :2])) < 2.0
