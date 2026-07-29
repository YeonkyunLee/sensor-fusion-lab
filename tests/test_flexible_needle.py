"""유연 바늘(exp 48) 테스트. 스캔 데이터가 없으면 일부 skip. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "flex48", ROOT / "scripts" / "48_flexible_needle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_beam_matches_analytic_cantilever():
    """조직 지지가 없으면 이산 보 해가 해석해 y=Fℓ³/(3EI) 와 일치해야 한다.

    이 검증이 실제로 버그를 잡았다: 꺾임각→처짐의 이중 적분을 한 번만 누적하면
    처짐이 수백 배 작게 나온다."""
    m = _load()
    L = 0.07
    _, y, _ = m.beam_deflection(L, k_t=0.0, n=200)
    analytic = m.F_BEVEL * L ** 3 / (3 * m.EI)
    assert abs(y[-1] - analytic) / analytic < 0.02, f"{y[-1]:.5f} vs {analytic:.5f}"


def test_tissue_support_reduces_deflection():
    """조직의 횡방향 지지는 처짐을 줄인다(같은 힘·같은 길이에서)."""
    m = _load()
    _, y_free, _ = m.beam_deflection(0.07, k_t=0.0)
    _, y_sup, _ = m.beam_deflection(0.07)
    assert 0 < y_sup[-1] < y_free[-1]


def test_deflection_scales_with_length_cubed():
    """외팔보 스케일링: 길이 2배 → 처짐 약 8배(지지 없음)."""
    m = _load()
    _, y1, _ = m.beam_deflection(0.04, k_t=0.0, n=200)
    _, y2, _ = m.beam_deflection(0.08, k_t=0.0, n=200)
    assert 7.0 < y2[-1] / y1[-1] < 9.0, y2[-1] / y1[-1]


def test_shape_integrates_curvature():
    """곡률 0 이면 직선, 곡률 κ 면 sagitta ≈ κℓ²/2."""
    m = _load()
    entry = np.zeros(3)
    axis = np.array([1.0, 0.0, 0.0])
    L = 0.07
    straight = m.needle_shape(entry, axis, L, 0.0)
    assert np.allclose(straight[-1], entry + L * axis, atol=1e-9)

    kappa = 3.0
    bent = m.needle_shape(entry, axis, L, kappa)
    sag = np.linalg.norm(bent[-1] - (entry + L * axis))
    assert abs(sag - kappa * L ** 2 / 2) / (kappa * L ** 2 / 2) < 0.1


def test_optimal_flip_depth_matches_analytic():
    """일정 곡률 두 호가 팁 편차를 정확히 상쇄하는 깊이는 (1 − 1/√2)·L ≈ 29.3%.

    2x² − 4x + 1 = 0 (x = d/L) 의 해. 50% 는 기울기만 상쇄하고 오프셋을 남기므로
    최적이 아니다 — 수치 스윕이 이 값을 재현하는지 확인한다."""
    m = _load()
    entry = np.zeros(3)
    axis = np.array([1.0, 0.0, 0.0])
    L, kappa = 0.07, 3.0

    depths = np.linspace(0.1, 0.9, 81) * L
    devs = [m.tip_deviation(m.needle_shape(entry, axis, L, kappa, m.policy_flip(d)),
                            entry, axis, L) for d in depths]
    d_best = depths[int(np.argmin(devs))] / L
    assert abs(d_best - (1 - 1 / np.sqrt(2))) < 0.03, f"최적 flip 깊이 {d_best:.3f}"


def test_spin_compensation_cancels_the_arc():
    """최적 깊이의 180° flip 과 duty cycling 모두 팁 편차를 크게 줄인다."""
    m = _load()
    entry = np.zeros(3)
    axis = np.array([1.0, 0.0, 0.0])
    L, kappa = 0.07, 3.0
    dev_plain = m.tip_deviation(m.needle_shape(entry, axis, L, kappa), entry, axis, L)
    dev_flip = m.tip_deviation(
        m.needle_shape(entry, axis, L, kappa, m.policy_flip((1 - 1 / np.sqrt(2)) * L)),
        entry, axis, L)
    dev_duty = m.tip_deviation(
        m.needle_shape(entry, axis, L, kappa, m.policy_duty(L / 3)), entry, axis, L)
    assert dev_flip < dev_plain / 10, f"flip {dev_flip*1e3:.2f} vs {dev_plain*1e3:.2f} mm"
    assert dev_duty < dev_plain / 3
    # 50% 뒤집기는 최적이 아니다(기울기만 상쇄)
    dev_half = m.tip_deviation(
        m.needle_shape(entry, axis, L, kappa, m.policy_flip(L / 2)), entry, axis, L)
    assert dev_half > dev_flip


def test_bending_eats_clearance_and_spin_restores_it():
    """전체 실행: 휨이 통로 여유를 갉아먹고, flip 보상이 되찾는다."""
    m = _load()
    archive = ROOT / m.g6.real.ARCHIVE
    if not archive.exists():
        try:
            m.g6.real.fetch(dest=archive)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Stanford Bunny 데이터 없음(오프라인): {e}")

    res = m.main()
    assert res["dev_bent"] > 1e-3, "휨이 사실상 없으면 실험 전제가 깨진다"
    assert res["cl_bent"] < res["cl_rigid"], "휨이 여유를 줄이지 않음"
    assert res["a_bent"] < res["a_rigid"], "휨이 축오차 임계를 낮추지 않음"
    # flip 보상이 팁 편차를 한 자릿수 이상 줄이고 여유를 되돌린다
    assert res["best_flip"][1] < res["dev_bent"] / 10
    assert res["a_flip"] > res["a_bent"]
