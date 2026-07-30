"""실 인체 MR 표면 정합(exp 49) 테스트. 데이터가 없고 못 받으면 skip.

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
        "anat49", ROOT / "scripts" / "49_registration_real_anatomy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_procrustes_recovers_a_known_rigid_transform():
    """랜드마크 조대정렬의 코어: 대응점에서 강체변환을 정확히 복원(반사 없이)."""
    m = _load()
    rng = np.random.default_rng(0)
    for _ in range(6):
        src = rng.normal(size=(5, 3)) * 0.05
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        from sensor_fusion.se3 import so3_exp
        R = so3_exp(ax * rng.uniform(-1.0, 1.0))
        t = rng.uniform(-0.05, 0.05, 3)
        dst = src @ R.T + t
        T = m.procrustes(src, dst)
        assert np.allclose(T[:3, :3], R, atol=1e-9)
        assert np.allclose(T[:3, 3], t, atol=1e-9)
        assert np.linalg.det(T[:3, :3]) > 0


def test_farthest_point_landmarks_are_spread_out():
    """랜드마크가 한쪽에 몰리면 조대정렬 회전이 부실해진다 — 퍼짐을 확인."""
    m = _load()
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(3000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)      # 구면
    idx = m.pick_landmarks(pts, k=4)
    d = np.linalg.norm(pts[idx][:, None] - pts[idx][None], axis=2)
    assert len(set(idx.tolist())) == 4
    assert d.max() > 1.5, f"랜드마크가 퍼지지 않음: {d.max():.2f}"


@pytest.fixture(scope="module")
def result():
    m = _load()
    nrrd = ROOT / m.MRHEAD
    if not nrrd.exists():
        try:
            m.fetch(dest=nrrd)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"공개 MR 데이터 없음(오프라인): {e}")
    return m, m.main(n_region=8, n_gate=12)


def test_volume_and_surface_are_anatomically_plausible(result):
    """실 인체 MR 로드 + 표면 추출: 머리 크기가 사람 규모여야 한다."""
    m, res = result
    vol, D, origin = m.load_nrrd()
    assert vol.ndim == 3 and min(vol.shape) > 50
    surface, _ = m.head_surface(vol, D, origin)
    assert len(surface) > 20000, f"표면 점군이 너무 적다: {len(surface)}"
    extent = (surface.max(0) - surface.min(0)) * 1e3
    assert np.all(extent > 100) and np.all(extent < 320), f"머리 크기 이상: {extent}"


def test_landmark_coarse_alignment_is_required(result):
    """국소 조각만으로는 무게중심 초기화가 실패하고, 랜드마크 조대정렬이 살린다."""
    _, res = result
    assert res["tre_centroid"] > 0.02, "무게중심 초기화가 실패하지 않으면 전제가 깨진다"
    assert res["tre_land"] < 3e-3, f"랜드마크 정합 TRE 과다: {res['tre_land']*1e3:.2f} mm"
    assert res["tre_centroid"] > 10 * res["tre_land"]


def test_registration_reaches_clinical_accuracy_on_real_anatomy(result):
    """표준 조건에서 TRE 중앙값이 임상 수준(≈2 mm 이내)에 들어온다."""
    _, res = result
    assert np.median(res["treA"]) < 2e-3, f"{np.median(res['treA'])*1e3:.2f} mm"


def test_probing_a_feature_rich_region_helps(result):
    """핵심 결과: 매끄러운 영역보다 특징적인 영역을 찍을 때 TRE 가 낮다."""
    _, res = result
    sv, tre = res["svA"], res["treA"]
    lo, hi = sv < np.median(sv), sv >= np.median(sv)
    assert np.median(tre[hi]) < np.median(tre[lo]), \
        f"특징적 영역 {np.median(tre[hi])*1e3:.2f} vs 매끄러운 {np.median(tre[lo])*1e3:.2f} mm"
    assert res["corr"] < 0, f"상관이 음수가 아니다: {res['corr']:+.2f}"


def test_verification_point_beats_the_covariance_gate(result):
    """임상 절차(검증점)가 공분산 게이트보다 실제 해부에서 잘 잡는다."""
    _, res = result
    assert res["det_verify"] >= res["det_sigma"], \
        f"검증점 {res['det_verify']:.0f}% vs σ {res['det_sigma']:.0f}%"
    assert res["det_verify"] > 80, f"검증점 검출력 부족: {res['det_verify']:.0f}%"
    assert res["corr_verify"] > 0.5, f"검증점↔TRE 상관 부족: {res['corr_verify']:+.2f}"


def test_spreading_the_probe_beats_concentrating_it(result):
    """같은 점 수라도 여러 영역에 나눠 찍는 편이 정확하다."""
    _, res = result
    cov = res["cov"]
    assert cov[4] < cov[1], f"흩어 찍기 이득 없음: {cov}"
