"""대응 탐색(exp 55) 테스트. 실 MR 데이터가 없고 못 받으면 skip.

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
        "corr55", ROOT / "scripts" / "55_correspondence_search.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 코어 — 구멍 문제의 기하
# --------------------------------------------------------------------------- #
def test_tangential_slide_does_not_move_a_smooth_surface():
    """실험의 전제: 매끄러운 표면을 접선 방향으로 밀면 **모양이 안 바뀐다**.

    그래서 최근접점이 그 몫을 볼 수 없다. 평면에서 극단적으로 확인한다."""
    m = _load()
    g = np.linspace(-0.05, 0.05, 160)                    # 촘촘한 격자여야 의미가 있다
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    slid = pts + np.array([3e-3, 0.0, 0.0])              # 평면 안에서 접선 이동
    d_tan, _ = tree.query(slid, k=1)
    lifted = pts + np.array([0.0, 0.0, 3e-3])            # 같은 크기의 법선 이동
    d_norm, _ = tree.query(lifted, k=1)
    assert d_tan.mean() < 0.2 * d_norm.mean(), \
        f"접선 {d_tan.mean()*1e3:.2f} vs 법선 {d_norm.mean()*1e3:.2f} mm — 전제가 깨진다"
    assert abs(d_norm.mean() - 3e-3) < 1e-9


def test_field_is_one_consistent_volumetric_field():
    """표면용/심부용 장을 따로 쓰면 '대응의 효과'가 아니라 '장 불일치'를 재게 된다.

    하나의 장이므로 표면점과 그 바로 안쪽 점의 변위가 연속이어야 한다."""
    m = _load()
    center = np.zeros(3)
    window_c = np.array([0.0, 0.0, 0.09])
    slide = np.array([1.0, 0.0, 0.0])
    p = window_c[None, :] + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1e-6]])
    u = m.field(p, center, window_c, slide)
    assert np.linalg.norm(u[0] - u[1]) < 1e-6, "장이 표면에서 불연속이다"


def test_split_on_surface_is_an_orthogonal_decomposition():
    """법선/접선 분해는 보고용이지만, 직교분해로서 정확해야 한다."""
    m = _load()
    rng = np.random.default_rng(1)
    n = rng.normal(size=(50, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    u = rng.normal(size=(50, 3)) * 1e-3
    un, ut = m.split_on_surface(u, n)
    assert np.allclose(un + ut, u, atol=1e-15)
    assert np.allclose(np.sum(ut * n, axis=1), 0.0, atol=1e-15)


def test_point_to_plane_keeps_only_the_normal_residual():
    """p2plane 은 접선 잔차를 **전부** 버린다 — 그게 나중에 대가가 되는 지점."""
    m = _load()
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(2)
    model = rng.normal(size=(300, 3)) * 0.03
    normals = rng.normal(size=(300, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    tree = cKDTree(model)
    obs = model + rng.normal(size=(300, 3)) * 1e-4
    ii, _, d = m.find_correspondence(obs, model, tree, "p2plane", normals)
    resid_tan = d - (np.sum(d * normals[ii], axis=1))[:, None] * normals[ii]
    assert np.allclose(resid_tan, 0.0, atol=1e-15), "p2plane 잔차에 접선 성분이 남아 있다"
    _, _, d_p2p = m.find_correspondence(obs, model, tree, "p2p", normals)
    assert np.linalg.norm(d_p2p) > np.linalg.norm(d), "p2p 는 접선 성분을 더 갖고 있어야 한다"


# --------------------------------------------------------------------------- #
# 전체 실험
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


def test_correspondence_error_tracks_the_slide_while_residual_does_not(result):
    """핵심 1 — 구멍 문제: 접선을 키우면 대응 오차만 늘고 표면 잔차는 그대로."""
    _, res = result
    A = res["A_rows"]                       # (설정, 실접선, 대응오차, 표면잔차, 법선회수율)
    assert A[-1][2] > 2.5 * A[0][2], f"대응 오차가 안 늘었다: {[a[2] for a in A]}"
    assert A[-1][3] < 1.6 * A[0][3], f"표면 잔차가 같이 늘면 관측 가능한 것: {[a[3] for a in A]}"
    assert A[-1][4] > 0.8, f"법선 성분 회수율 {A[-1][4]:.2f} — 보이는 몫은 잡아야 한다"


def test_finding_correspondence_costs_a_real_multiple(result):
    """핵심 2 — exp 51~54 가 '주어진 것'으로 두던 몫을 숫자로."""
    _, res = result
    assert res["e_p2p"] > 1.8 * res["e_gt"], \
        f"최근접점 {res['e_p2p']*1e3:.2f} vs 정답 {res['e_gt']*1e3:.2f} mm"
    assert res["e_gt"] < 1.0e-3, "정답 대응에서 exp 51 수준(≈0.6 mm)이 안 나오면 설정이 틀렸다"


def test_point_to_plane_is_not_a_free_win(result):
    """정직한 네거티브: 접선 잔차를 전부 버리는 것이 여기서는 손해다."""
    _, res = result
    assert res["e_p2pl"] > res["e_p2p"], \
        f"p2plane {res['e_p2pl']*1e3:.2f} vs p2p {res['e_p2p']*1e3:.2f} mm"
    assert res["s_p2pl"] < res["s_p2p"], "그런데 표면 잔차는 더 작다 — 지표가 또 배신한다"


def test_a_handful_of_landmarks_does_not_fix_a_field_wide_bias(result):
    """정직한 네거티브: 앵커 2~16개로는 창 전체에 깔린 편향을 못 덮는다."""
    _, res = result
    best = min(b[1] for b in res["B_rows"] if b[0])
    assert best > 0.6 * res["e_p2p"], \
        f"랜드마크가 {best*1e3:.2f} mm 로 크게 고쳐지면 결론이 달라진다"


def test_the_gain_is_roughly_linear_in_the_correspondence_fraction(result):
    """핵심 3 — 싼 해법이 없다: 비기하학적 대응 비율에 거의 선형."""
    _, res = result
    F = res["F_rows"]
    errs = [f[1] for f in F]
    assert errs[0] > errs[-1], "비율을 올리면 좋아져야 한다"
    assert abs(errs[0] - res["e_p2p"]) < 1e-9, "비율 0% 는 최근접점과 같은 시드여야 한다"
    assert abs(errs[-1] - res["e_gt"]) < 0.1e-3, "비율 100% 는 정답 대응과 같아야 한다"
    mid = next(e for f, e in F if abs(f - 0.25) < 1e-9)
    linear = errs[0] + 0.25 * (errs[-1] - errs[0])
    assert abs(mid - linear) < 0.35e-3, \
        f"25% 에서 {mid*1e3:.2f} vs 선형 기대 {linear*1e3:.2f} mm — 급격한 이득이면 결론이 바뀐다"


def test_robust_barely_helps_because_the_outliers_do_not_look_like_outliers(result):
    """정직한 네거티브: 최근접점이 튄 관측점 근처의 다른 표면점을 찾아 주므로, 대응은 틀렸는데
    변위는 작고 그럴듯하다 — 잔차 기반 로버스트가 볼 것이 없다."""
    _, res = result
    e_ls, e_rb = res["outlier"]
    assert e_rb < 1.2 * e_ls, f"로버스트가 크게 해로우면 안 된다: {e_rb*1e3:.2f} vs {e_ls*1e3:.2f} mm"
    assert min(e_ls, e_rb) > 1.8 * res["e_gt"], \
        "어느 쪽이든 정답 대응 수준에 못 간다 — 접선 몫은 로버스트의 문제가 아니다"
