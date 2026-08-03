"""비이상 초음파 모델(exp 53) 테스트. 실 MR 데이터가 없고 못 받으면 skip.

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
        "meas53", ROOT / "scripts" / "53_measurement_changes_it.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 데이터 없이 도는 코어 테스트
# --------------------------------------------------------------------------- #
def test_indentation_is_always_inward_and_decays():
    """압박 편향의 핵심 성질: 방향이 **늘 안쪽**이고 거리로 감쇠한다.

    부호가 늘 같다는 것이 '평균되지 않는다'의 이유이므로, 이게 깨지면 실험 전제가 무너진다."""
    m = _load()
    inward = np.array([0.2, -0.3, -1.0]); inward /= np.linalg.norm(inward)
    contact = np.array([0.01, 0.02, 0.05])
    pts = contact + np.linspace(0, 0.12, 40)[:, None] * inward[None, :]
    u = m.indentation_field(pts, contact, inward, 3.5e-3)
    assert np.all(u @ inward >= -1e-15), "안쪽 성분이 음수인 지점이 있다"
    mag = np.linalg.norm(u, axis=1)
    assert np.all(np.diff(mag) <= 1e-15), "거리에 따라 단조 감소해야 한다"
    assert abs(mag[0] - 3.5e-3) < 1e-12, "접촉점에서 압박 깊이와 같아야 한다"


def test_indentation_does_not_average_out_over_contact_points():
    """접촉점을 여기저기 옮겨도 평균이 0으로 가지 않는다 — 계통 오차의 정의."""
    m = _load()
    rng = np.random.default_rng(0)
    inward = np.array([0.0, 0.0, -1.0])
    target = np.array([0.0, 0.0, -0.045])
    acc = []
    for _ in range(400):
        contact = np.array([rng.normal(0, 0.02), rng.normal(0, 0.02), 0.0])
        acc.append(m.indentation_field(target[None], contact, inward,
                                       rng.uniform(*m.INDENT_MM) * 1e-3)[0])
    mean = np.mean(acc, axis=0)
    assert np.linalg.norm(mean) > 0.3e-3, \
        f"400회 평균이 {np.linalg.norm(mean)*1e3:.3f} mm — 상쇄되면 실험 전제가 깨진다"
    assert float(mean @ inward) > 0, "평균 편향이 안쪽을 향해야 한다"


def test_depth_sigma_grows_with_depth():
    """가장 정보가 필요한 깊은 곳이 가장 못 믿을 곳이라는 전제."""
    m = _load()
    window_c = np.zeros(3)
    inward = np.array([0.0, 0.0, -1.0])
    pts = window_c + np.array([0.0, 0.02, 0.05, 0.09])[:, None] * inward[None, :]
    s = m.depth_sigma(pts, window_c, inward)
    assert abs(s[0] - m.US_NOISE) < 1e-12, "표면에서는 σ₀ 여야 한다"
    assert np.all(np.diff(s) > 0)
    assert s[-1] > 2 * s[0]


def test_contact_point_sits_on_the_exposed_surface_plane():
    """프로브는 보고 싶은 곳 **바로 위**를 누른다 — 접촉점의 깊이 성분은 0."""
    m = _load()
    window_c = np.array([0.01, -0.02, 0.05])
    inward = np.array([0.3, -0.2, -1.0]); inward /= np.linalg.norm(inward)
    pts = m.probe.cone_points(np.random.default_rng(2), 50, window_c, inward)
    c = m.contact_for(pts, window_c, inward)
    assert np.allclose((c - window_c) @ inward, 0.0, atol=1e-12)


def test_robust_fit_beats_least_squares_with_outliers():
    """Huber IRLS 의 코어 — 데이터 없이 합성 문제로 확인."""
    m = _load()
    rng = np.random.default_rng(3)
    ctrl = rng.uniform(-0.05, 0.05, (80, 3))
    true = 2e-3 * np.sin(ctrl * 40.0)
    obs = true + rng.normal(0, 1e-4, true.shape)
    bad = rng.random(len(ctrl)) < 0.2
    obs[bad] += rng.normal(0, 2e-2, (int(bad.sum()), 3))
    lam = np.full(len(ctrl), 1e-3)
    q = rng.uniform(-0.04, 0.04, (200, 3))
    ref = 2e-3 * np.sin(q * 40.0)
    e_ls = np.linalg.norm(m.deform.tps_apply(
        m.deform.tps_fit(ctrl, obs, lam=lam), q) - q - ref, axis=1).mean()
    e_rb = np.linalg.norm(m.deform.tps_apply(
        m.robust_tps_fit(ctrl, obs, lam), q) - q - ref, axis=1).mean()
    assert e_rb < 0.7 * e_ls, f"로버스트 이득 없음: {e_rb:.2e} vs {e_ls:.2e}"


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
    return m, m.main(quick=True, n_trials=240)


def test_random_noise_averages_out_but_bias_does_not(result):
    """핵심 1: 이상 센서 곡선은 계속 내려가고, 압박이 있으면 **격차가 유지**된다."""
    _, res = result
    A, n = res["A"], res["n_list"]
    assert A["ideal"][-1] < 0.6 * A["ideal"][0], "이상 센서에서 관측이 도움이 안 되면 전제가 깨진다"
    gap0 = A["press"][0] - A["ideal"][0]
    gapN = A["press"][-1] - A["ideal"][-1]
    assert gap0 > 0 and gapN > 0, "압박이 오차를 늘려야 한다"
    assert gapN > 0.5 * gap0, \
        f"격차가 사라졌다({gap0*1e3:.2f} → {gapN*1e3:.2f} mm): 계통 오차가 평균되면 안 된다"
    share0 = gap0 / A["press"][0]
    shareN = gapN / A["press"][-1]
    assert shareN > share0, \
        f"관측을 늘릴수록 편향의 몫이 커져야 한다: {share0:.2f} → {shareN:.2f}"


def test_de_indentation_recovers_but_needs_the_right_model(result):
    """압박은 모델로 뺄 수 있다 — 단 모델을 알아야 하고, 틀린 만큼 잔차가 남는다."""
    _, res = result
    A = res["A"]
    assert A["deind_ok"][-1] < A["press"][-1], "정확한 보정이 압박을 줄여야 한다"
    assert A["deind_ok"][-1] <= A["deind_err"][-1] + 1e-9, \
        "정확한 모델이 틀린 모델보다 나쁠 수 없다"
    assert A["deind_ok"][-1] < 1.15 * A["ideal"][-1], \
        f"정확히 보정하면 이상 센서 수준({A['ideal'][-1]*1e3:.2f})에 근접해야 한다"


def test_depth_weighting_helps_but_does_not_restore(result):
    """가중은 정보를 만들지 못한다 — 이득은 있지만 등방 잡음 수준으로 돌아가지 못한다."""
    _, res = result
    B = res["B"]
    assert B["depth"][-1] > B["flat"][-1], "σ(d) 가 오차를 늘려야 한다"
    assert B["weighted"][-1] < B["depth"][-1], "올바른 가중이 도움이 되어야 한다"
    assert B["weighted"][-1] > B["flat"][-1], \
        "가중만으로 등방 잡음 수준을 회복하면 '정보를 만들지 못한다'는 주장이 깨진다"


def test_outliers_break_least_squares_and_robust_recovers(result):
    """exp 11·15 의 로버스트 커널이 정합에서 재사용된다."""
    _, res = result
    C = res["C"]
    assert C["ls"][-1] > 1.5 * C["clean"][-1], \
        f"오대응이 최소제곱을 망가뜨리지 않았다: {C['ls'][-1]*1e3:.2f} vs {C['clean'][-1]*1e3:.2f}"
    # 관측이 적으면(1~4개) 이상치 하나를 버리는 대가가 커서 로버스트가 손해일 수 있다 —
    # 그래서 관측이 충분한 끝단에서만 이득을 요구한다.
    assert C["robust"][-1] < C["ls"][-1], "관측이 충분한데도 로버스트 이득이 없다"
    gap = C["ls"][-1] - C["clean"][-1]
    recovered = (C["ls"][-1] - C["robust"][-1]) / gap
    assert recovered > 0.25, f"로버스트가 되찾은 몫이 {recovered*100:.0f}% 뿐"


def test_the_gate_degrades_when_the_check_uses_the_same_sensor(result):
    """핵심 2: exp 52 의 게이트 숫자는 검산까지 이상 센서였기에 나온 값이다."""
    _, res = result
    s = res["summary"]
    ideal = max(s["ideal"]["auroc"].values())
    realistic = max(max(s["real"]["auroc"].values()),
                    max(s["real_fix"]["auroc"].values()))
    assert ideal > realistic + 0.10, \
        f"이상 {ideal:.2f} vs 현실 {realistic:.2f} — 차이가 없으면 이 실험의 요점이 없다"
    assert realistic > 0.52, "그래도 표면 게이트(동전)보다는 나아야 한다"


def test_remedies_fix_the_correction_not_the_gate(result):
    """대책은 자기가 겨냥한 오차원만 고친다 — 편향·이상치 대책이 잡음 한계를 못 넘는다."""
    _, res = result
    s = res["summary"]
    assert s["real_fix"]["fixed"] < 0.8 * s["real"]["fixed"], \
        f"교정이 회복되지 않았다: {s['real']['fixed']*1e3:.2f} → {s['real_fix']['fixed']*1e3:.2f} mm"
    gain = max(s["real_fix"]["auroc"].values()) - max(s["real"]["auroc"].values())
    assert gain < 0.15, \
        f"게이트까지 크게 회복되면 '잡음에 묶여 있다'는 설명이 틀린 것: {gain:+.2f}"


def test_main_is_reproducible(result):
    """문자열 hash 를 시드 salt 로 쓰면 실행마다 결과가 바뀐다(실제로 그렇게 짰다가 잡았다).

    같은 인자로 두 번 부르면 같은 숫자가 나와야 한다."""
    m, res = result
    again = m.main(quick=True, n_trials=240)
    for k in ("ideal", "real", "real_fix"):
        assert again["summary"][k]["auroc"] == res["summary"][k]["auroc"], \
            f"{k} 게이트가 재현되지 않는다"
    assert again["A"]["press"] == res["A"]["press"]
