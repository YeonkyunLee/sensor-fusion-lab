"""변형 정합(exp 51) 테스트. 실 MR 데이터가 없고 못 받으면 skip.

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
        "deform51", ROOT / "scripts" / "51_deformable_registration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 데이터 없이 도는 코어 테스트
# --------------------------------------------------------------------------- #
def test_tps_interpolates_its_control_points():
    """λ→0 이면 TPS 는 제어점을 정확히 통과해야 한다(보간 조건)."""
    m = _load()
    rng = np.random.default_rng(0)
    ctrl = rng.uniform(-0.05, 0.05, (40, 3))
    disp = rng.normal(0, 2e-3, (40, 3))
    model = m.tps_fit(ctrl, disp, lam=0.0)
    got = m.tps_apply(model, ctrl) - ctrl
    assert np.allclose(got, disp, atol=1e-9), f"보간 잔차 {np.abs(got - disp).max():.2e}"


def test_tps_reproduces_an_affine_field_exactly():
    """아핀 항이 있으므로 균일한 변위·선형 변위는 오차 없이 재현되어야 한다."""
    m = _load()
    rng = np.random.default_rng(1)
    ctrl = rng.uniform(-0.05, 0.05, (30, 3))
    A, b = rng.normal(0, 0.05, (3, 3)), rng.normal(0, 2e-3, 3)
    model = m.tps_fit(ctrl, ctrl @ A.T + b, lam=1e-6)
    q = rng.uniform(-0.08, 0.08, (25, 3))                # 제어점 밖 = 외삽
    assert np.allclose(m.tps_apply(model, q) - q, q @ A.T + b, atol=1e-6)


def test_tps_regularization_is_monotone_toward_affine():
    """λ 를 키우면 워프가 아핀에 가까워진다 — 부호를 잘못 쓰면 이 단조성이 깨진다.

    (처음에 커널을 φ(r)=+r 로 짰다가 λ 스윕이 비단조로 폭주했던 실제 버그를 잡는 테스트.)"""
    m = _load()
    rng = np.random.default_rng(2)
    ctrl = rng.uniform(-0.05, 0.05, (60, 3))
    disp = 3e-3 * np.sin(ctrl * 60.0)                    # 비아핀 성분이 큰 장
    q = rng.uniform(-0.05, 0.05, (200, 3))
    prev = None
    for lam in (1e-6, 1e-3, 1e-1, 1e1):
        model = m.tps_fit(ctrl, disp, lam=lam)
        w = m.tps_apply(model, q) - q
        lin = np.hstack([np.ones((len(q), 1)), q])
        nonaffine = float(np.linalg.norm(w - lin @ np.linalg.lstsq(lin, w, rcond=None)[0]))
        if prev is not None:
            assert nonaffine <= prev * 1.02, f"λ={lam:g} 에서 비아핀 성분이 증가"
        prev = nonaffine


def test_deformation_field_decays_away_from_the_craniotomy():
    """변형장 전제: 개두창에서 최대이고 멀어지면 사라진다(두개골이 잡는다)."""
    m = _load()
    wc = np.zeros(3)
    sag = np.array([0.0, 0.0, -1.0])
    inw = np.array([0.0, 0.0, -1.0])
    d = np.linspace(0.0, 0.20, 30)
    pts = wc + d[:, None] * np.array([1.0, 0.0, 0.0])
    mag = np.linalg.norm(m.deformation(pts, wc, sag, inw), axis=1)
    assert np.all(np.diff(mag) < 1e-12), "거리에 따라 단조 감소해야 한다"
    assert mag[0] > 10e-3 and mag[-1] < 1e-3, f"{mag[0]*1e3:.1f} → {mag[-1]*1e3:.2f} mm"


# --------------------------------------------------------------------------- #
# 실 MR 데이터 기반
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def result():
    m = _load()
    nrrd = ROOT / m.anat.MRHEAD
    if not nrrd.exists():
        try:
            m.anat.fetch(dest=nrrd)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"공개 MR 데이터 없음(오프라인): {e}")
    return m, m.main(quick=True)


def test_rigid_registration_is_accurate_before_the_deformation(result):
    """단계 1 전제: 개두 전 강체 정합 자체는 exp 49 수준으로 정확하다.

    이게 성립해야 이후 오차를 전부 '변형' 탓으로 돌릴 수 있다."""
    _, res = result
    assert res["rigid_only_mm"] < 1.0, f"강체 정합이 이미 부정확: {res['rigid_only_mm']:.2f} mm"


def test_rigid_registration_leaves_the_whole_deformation_at_the_target(result):
    """강체는 변형을 원리적으로 못 잡는다 — 표적 오차 ≈ 그 지점의 실제 변형량."""
    _, res = result
    rigid = res["per_depth"]["rigid"]
    assert np.allclose(rigid, res["u_target"], rtol=0.25), \
        f"강체 오차 {rigid*1e3} vs 변형량 {res['u_target']*1e3} mm"


def test_deformable_recovers_most_of_the_shift(result):
    """변형 복원의 존재 이유: 강체가 남긴 mm 를 실제로 줄인다."""
    _, res = result
    for row in res["rows"]:
        best = min(row["tre"][k] for k in ("tps", "tps_prior", "harmonic"))
        assert best < 0.6 * row["tre"]["rigid"], \
            f"노출 {row['exposure']:.0f}°: {best*1e3:.2f} vs 강체 {row['tre']['rigid']*1e3:.2f} mm"


def test_the_prior_not_the_interpolator_wins_at_narrow_exposure(result):
    """핵심 결과: 좁은 노출에서는 같은 보간기 + 사전지식이 크게 이긴다."""
    _, res = result
    tight = res["rows"][0]
    assert tight["tre"]["tps_prior"] < 0.5 * tight["tre"]["tps"], \
        f"사전지식 이득 없음: {tight['tre']['tps_prior']*1e3:.2f} vs {tight['tre']['tps']*1e3:.2f} mm"


def test_the_prior_stops_paying_when_data_is_plentiful(result):
    """반대편도 확인: 넓은 노출에서는 사전지식이 할 일이 없다(이득이 사라진다)."""
    _, res = result
    wide = res["rows"][-1]
    ratio = wide["tre"]["tps"] / wide["tre"]["tps_prior"]
    assert 0.7 < ratio < 1.4, f"넓은 노출에서도 사전지식 효과가 크다: {ratio:.2f}배"


def test_finer_grid_makes_the_harmonic_model_worse(result):
    """모델 편향의 증거: 격자를 촘촘히 할수록 더 정확히 **틀린** 답으로 수렴한다."""
    _, res = result
    g = res["grid_rows"]
    assert len(g) >= 2
    assert g[-1][3] > g[0][3], f"격자를 촘촘히 했는데 나빠지지 않음: {[x[3] for x in g]}"
    assert g[-1][4] < g[0][4] < 1.0, f"깊이 변위 예측 비율: {[x[4] for x in g]}"


def test_optimal_degrees_of_freedom_move_with_exposure(result):
    """자유도(제어점 수)의 최적점이 노출 면적에 따라 이동한다 — 고정하면 손해."""
    m, res = result
    c = res["ctrl_rows"]
    b45 = m.N_CTRL_SWEEP[int(np.argmin(c[45.0]))]
    b110 = m.N_CTRL_SWEEP[int(np.argmin(c[110.0]))]
    assert b110 > b45, f"넓은 노출이 더 많은 자유도를 요구하지 않음: {b45} vs {b110}"


def test_surface_residual_does_not_rank_methods_by_deep_accuracy(result):
    """FRE≠TRE 의 변형판: 볼 수 있는 지표(표면 잔차)의 순위가 심부 순위와 다르다."""
    _, res = result
    methods = ("tps", "tps_prior", "harmonic")
    mismatch = 0
    for row in res["rows"]:
        by_srf = min(methods, key=lambda k: row["srf"][k])
        by_tre = min(methods, key=lambda k: row["tre"][k])
        mismatch += by_srf != by_tre
    assert mismatch > 0, "모든 노출에서 순위가 일치하면 이 실험의 전제가 깨진다"
