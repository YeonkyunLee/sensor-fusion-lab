"""실 스캔 정합 검증(exp 44) 테스트. 데이터가 없고 내려받지도 못하면 skip.

실행: pytest -q
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "real44", ROOT / "scripts" / "44_registration_real_scans.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bunny():
    """Stanford Bunny 를 준비한다. 오프라인이면 이 모듈 전체를 skip."""
    mod = _load()
    archive = ROOT / mod.ARCHIVE
    if not archive.exists():
        try:
            mod.fetch(dest=archive)
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Stanford Bunny 데이터 없음(오프라인): {e}")
    scans = mod.load_bunny(archive=archive)
    rng = np.random.default_rng(0)
    full0 = scans["bun000"]
    sel = rng.choice(len(full0), mod.MODEL_SUBSAMPLE, replace=False)
    model = full0[sel]
    return mod, scans, model, mod.reg.estimate_normals(model, k=12), cKDTree(model)


def test_scans_parse_and_conf_alignment_is_consistent(bunny):
    """PLY·conf 파싱 검증: conf 정렬 후 두 실 스캔이 서브밀리미터로 겹친다."""
    mod, scans, *_ = bunny
    full0, full45 = scans["bun000"], scans["bun045"]
    assert len(full0) > 30000 and len(full45) > 30000

    d, _ = cKDTree(full0).query(full45)
    # 규약이 틀리면(예: R@p+t) 중앙값이 수십 mm 로 벌어진다
    assert np.median(d) < 1e-3, f"conf 정렬 실패: 중앙값 {np.median(d)*1e3:.1f} mm"


def test_registration_recovers_pose_on_real_scans(bunny):
    """exp 41 파이프라인이 실 스캔에서도 미지 SE(3)를 서브밀리미터로 복원."""
    mod, scans, model, normals, tree = bunny
    targets = mod.targets_inside(model)
    rng = np.random.default_rng(3)
    T_pert = mod.random_perturb(rng)
    probe = mod.make_probe(rng, scans["bun045"], T_pert)

    T_reg, fre, sig, dis, inlier = mod.register(probe, model, normals, tree, targets[0])
    comp = T_reg @ T_pert
    rot = np.rad2deg(np.arccos(np.clip((np.trace(comp[:3, :3]) - 1) / 2, -1, 1)))
    tre = float(np.mean(mod.tre_of(T_reg, T_pert, targets)))

    assert rot < 0.5, f"회전 복원 오차 과다: {rot:.2f}°"
    assert np.linalg.norm(comp[:3, 3]) < 1e-3
    assert tre < 5e-4, f"실 스캔 TRE 과다: {tre*1e3:.2f} mm"
    assert inlier > 0.7


def test_more_probed_points_improve_tre_on_real_data(bunny):
    """커버리지 효과가 실측 기하에서도 재현된다."""
    mod, scans, model, normals, tree = bunny
    targets = mod.targets_inside(model)

    def med_tre(n):
        vals = []
        for s in range(3):
            r = np.random.default_rng(200 + s)
            Tp = mod.random_perturb(r)
            pr = mod.make_probe(r, scans["bun045"], Tp, n=n)
            Tr = mod.register(pr, model, normals, tree, targets[0], multistart=False)[0]
            vals.append(float(np.mean(mod.tre_of(Tr, Tp, targets))))
        return float(np.median(vals))

    assert med_tre(2000) < med_tre(150)


def test_consistency_signal_transfers_better_than_covariance(bunny):
    """핵심 전이 결과: 합성에서 통하던 k·σ 보다 다중초기값 일치성이 실 스캔에서 낫다."""
    mod, scans, model, normals, tree = bunny
    targets = mod.targets_inside(model)
    vt = targets[0]

    sig, dis, unsafe = [], [], []
    for i in range(12):
        r = np.random.default_rng([7, i])
        Tp = mod.random_perturb(r)
        patch = (int(r.integers(len(scans["bun045"]))), float(r.uniform(0.05, 0.25))) \
            if i % 2 == 0 else None
        pr = mod.make_probe(r, scans["bun045"], Tp, patch=patch)
        Tr, _, s, d, _ = mod.register(pr, model, normals, tree, vt)
        sig.append(s)
        dis.append(d)
        unsafe.append(float(np.mean(mod.tre_of(Tr, Tp, targets))) > mod.MISS_TOL)

    sig, dis, unsafe = np.array(sig), np.array(dis), np.array(unsafe)
    if unsafe.sum() == 0:
        pytest.skip("이 시드에서는 실패 사례가 생기지 않음")

    det_sigma = np.sum(unsafe & (mod.K_SIGMA * sig > mod.MISS_TOL)) / unsafe.sum()
    det_dis = np.sum(unsafe & (dis > mod.DISAGREE_TOL)) / unsafe.sum()
    assert det_dis > det_sigma, f"일치성 {det_dis:.2f} vs σ {det_sigma:.2f}"
    assert det_dis >= 0.5, f"일치성 검출력 부족: {det_dis:.2f}"
