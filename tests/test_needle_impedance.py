"""조직 반력·임피던스 제어(exp 47) 테스트. 스캔 데이터가 없으면 skip.

실행: pytest -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "imp47", ROOT / "scripts" / "47_needle_impedance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    m = _load()
    archive = ROOT / m.g6.real.ARCHIVE
    if not archive.exists():
        try:
            m.g6.real.fetch(dest=archive)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Stanford Bunny 데이터 없음(오프라인): {e}")
    return m


def test_tissue_model_has_the_three_phases():
    """접촉 전 0 → 누를수록 증가 → 임계 초과 시 관통하며 급락 → 이후 마찰."""
    m = _load()
    tissue = m.NeedleTissue(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    assert np.allclose(tissue.force(np.array([0.0, 0.0, -0.01])), 0.0)  # 접촉 전

    f_small = np.linalg.norm(tissue.force(np.array([0.0, 0.0, 0.002])))
    f_big = np.linalg.norm(tissue.force(np.array([0.0, 0.0, 0.005])))
    assert 0 < f_small < f_big, "누를수록 커져야 한다"
    assert not tissue.punctured

    # 충분히 깊게 누르면 관통하고 힘이 급락한다
    f_punc = np.linalg.norm(tissue.force(np.array([0.0, 0.0, 0.02])))
    assert tissue.punctured
    assert f_punc < m.F_PUNCTURE, f"관통 후에도 힘이 크다: {f_punc}"
    assert tissue.puncture_k is not None


def test_lateral_resistance_pulls_the_needle_back_to_axis():
    """축에서 벗어나면 조직이 되돌리는 방향으로 힘을 준다."""
    m = _load()
    tissue = m.NeedleTissue(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    f = tissue.force(np.array([0.003, 0.0, 0.001]))
    assert f[0] < 0, "축에서 +x 로 벗어났으면 −x 로 되밀어야 한다"


def test_apparent_inertia_is_symmetric_positive_definite(mod):
    """작업공간 유효관성 Λ = (J M⁻¹ Jᵀ)⁻¹ 의 기본 성질."""
    m = mod
    q = np.array([0.6, -1.15, 1.25, -1.05, -1.55, 0.0])
    Lam, J = m.apparent_inertia(q)
    assert np.allclose(Lam, Lam.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(Lam) > 0)
    # 병진 방향 유효질량은 팔 질량 규모(수 kg)여야 한다
    assert 1.0 < np.trace(Lam[:3, :3]) / 3 < 50.0


def test_position_control_is_stiffer_impedance_yields(mod):
    """핵심 비교: 위치 제어는 정확하지만 뻣뻣하고, 임피던스는 양보한다."""
    m = mod
    task = m.build_task(dt=m.DT)
    pos = m.run_case(task, "position")
    imp = m.run_case(task, "impedance", kp_task=100.0)

    assert pos["ok"] and imp["ok"], "시뮬레이션이 발산했다"
    assert pos["punctured"] and imp["punctured"]
    assert imp["final_err"] > 2 * pos["final_err"], "임피던스가 양보하지 않음"
    # 관통 돌진은 뻣뻣한 쪽이 더 크다
    assert pos["lunge"] > imp["lunge"]


def test_stiffness_sets_the_tradeoff(mod):
    """강성을 올리면 정확해지고, 관통 돌진은 커진다(단조 트레이드오프)."""
    m = mod
    task = m.build_task(dt=m.DT)
    soft = m.run_case(task, "impedance", kp_task=100.0)
    stiff = m.run_case(task, "impedance", kp_task=400.0)
    assert stiff["final_err"] < soft["final_err"]
    assert stiff["lunge"] >= soft["lunge"]


def test_interaction_term_enters_the_error_budget(mod):
    """접촉이 들어오면 상호작용 오차가 정합 몫(exp 45: 0.081 mm)을 넘어선다."""
    m = mod
    task = m.build_task(dt=m.DT)
    pos = m.run_case(task, "position")
    assert pos["final_err"] > 0.081e-3, "접촉인데 자유공간과 다를 바 없다"
    assert pos["final_err"] < 5e-3, "위치 제어인데 오차가 비현실적으로 크다"
