"""6-DOF 영상유도 유도(exp 45) 테스트. 스캔 데이터가 없고 못 받으면 skip.

실행: pytest -q
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sensor_fusion.se3 import so3_exp  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location(
        "guide45", ROOT / "scripts" / "45_image_guided_6dof.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    m = _load()
    archive = ROOT / m.real.ARCHIVE
    if not archive.exists():
        try:
            m.real.fetch(dest=archive)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Stanford Bunny 데이터 없음(오프라인): {e}")
    return m


def test_frame_from_axis_aligns_tool_z():
    """도구 자세 생성기: z축이 지정한 삽입축과 정확히 정렬되고 회전행렬이 정규직교."""
    m = _load()
    rng = np.random.default_rng(0)
    for _ in range(6):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = m.frame_from_axis(axis, roll=rng.uniform(0, 6.28))
        assert np.allclose(R[:, 2], axis, atol=1e-12)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) > 0


def test_rotation_between_maps_vectors():
    m = _load()
    rng = np.random.default_rng(1)
    for _ in range(6):
        a, b = rng.normal(size=3), rng.normal(size=3)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        assert np.allclose(m.rotation_between(a, b) @ a, b, atol=1e-9)


def test_shaft_clearance_sees_what_tip_check_cannot():
    """핵심 6-DOF 논점: 팁은 여유가 있는데 몸통이 관통하는 자세가 존재한다."""
    m = _load()
    tip = np.zeros(3)
    axis = np.array([0.0, 0.0, 1.0])          # 몸통은 tip 에서 −z 방향으로 뻗는다
    vessel = np.array([0.004, 0.0, -0.03])    # 팁에서 30 mm 뒤, 옆으로 4 mm
    tip_clear = np.linalg.norm(tip - vessel) - m.VESSEL_R
    assert tip_clear > 0                       # 팁 기준으로는 통과
    assert m.shaft_clearance(tip, axis, vessel) < 0   # 몸통은 관통


def test_plan_creates_a_tight_but_feasible_corridor(mod):
    """계획된 통로는 통과 가능하되 빡빡해야(안전 논의가 의미 있으려면) 한다."""
    m = mod
    m.real.fetch()
    scans = m.real.load_bunny()
    rng = np.random.default_rng(0)
    full0 = scans["bun000"]
    sel = rng.choice(len(full0), m.real.MODEL_SUBSAMPLE, replace=False)
    model = full0[sel]

    target, vessel, entry, axis, clear = m.plan_in_phantom(model)
    assert 0.0 < clear < 0.01, f"통로 여유 {clear*1e3:.1f} mm"
    assert np.linalg.norm(target - entry) > 0.03
    assert np.allclose(np.linalg.norm(axis), 1.0)
    # 삽입축은 진입점→표적 방향
    assert np.allclose(axis, (target - entry) / np.linalg.norm(target - entry), atol=1e-9)


def test_full_pipeline_registration_dominates_and_control_is_stable(mod):
    """전체 실행: 정합 서브밀리미터, IK 잔차 무시가능, 계산토크 서보 몫 ≪ 정합 몫."""
    res = mod.main(n_safety=8)

    assert res["tre"] < 1e-3, f"6-DOF TRE {res['tre']*1e3:.2f} mm"
    assert res["rot_err"] < 1.0
    assert res["ik_p"] < 1e-5 and res["ik_r"] < 1e-5
    assert res["w_min"] > 1e-2, "경로가 손목 특이점에 너무 가깝다"

    ct = res["out"]["ct"]
    pdg = res["out"]["pdg"]
    assert not ct["diverged"]
    assert ct["servo_pos"] < res["tre"], "보정된 계산토크에선 정합이 지배해야 한다"
    assert ct["servo_pos"] < pdg["servo_pos"] / 10, "모델기반 제어의 이득이 없음"
    assert res["out"]["pd"]["servo_pos"] > 1e-2, "PD 는 중력에 눌려 크게 처져야 한다"
