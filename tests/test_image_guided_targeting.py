"""영상유도 도구 유도 캡스톤(exp 42) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "guide42", ROOT / "scripts" / "42_image_guided_targeting.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registration_recovers_patient_pose():
    """부분 커버리지·잡음·outlier가 있어도 ICP가 환자 자세를 mm 이하로 복원한다."""
    mod = _load()
    model, normals = mod.build_model()
    rng = np.random.default_rng(0)
    T_true = mod.se2(np.deg2rad(9.0), np.array([0.43, 0.05]))
    probe = mod.sample_probe(rng, T_true, np.deg2rad(230), phi0=np.deg2rad(20))

    T_reg, fre, Cov, disagree = mod.register(probe, model, normals)
    # T_reg(robot→image) 와 T_true(image→robot)의 합성은 항등이어야 한다
    tre = np.linalg.norm(mod.apply_se2(T_reg @ T_true, mod.TUMOR_IMG) - mod.TUMOR_IMG)
    assert tre < 1e-3, f"TRE 과다: {tre*1e3:.2f} mm"
    assert fre < 2 * mod.PROBE_NOISE, f"FRE가 잡음 바닥 대비 과다: {fre*1e3:.2f} mm"
    assert disagree < mod.DISAGREE_TOL, "좋은 정합인데 다중초기값 해가 불일치"
    assert np.all(np.isfinite(Cov))


def test_ik_and_control_track_the_planned_path():
    """계획 경로 → IK → 계산토크 추종: IK 잔차·서보 오차 모두 무시할 수준."""
    mod = _load()
    T = mod.se2(np.deg2rad(8.0), np.array([0.43, 0.05]))
    entry, _ = mod.plan_entry()
    r = mod.run_condition(entry, mod.TUMOR_IMG, T, T, mod.dyn.ctrl_computed_torque)

    assert r["ik_worst"] < 1e-5, f"IK 잔차 과다: {r['ik_worst']*1e6:.1f} µm"
    assert r["total"] < 1e-4, f"완전정합인데 표적오차 과다: {r['total']*1e3:.3f} mm"
    assert r["insert_dev"] < 1e-4, f"삽입 경로이탈 과다: {r['insert_dev']*1e3:.3f} mm"


def test_registration_dominates_the_error_budget_when_calibrated():
    """오차 예산: 정합 없음 ≫ PD ≫ 계산토크. 보정된 팔에서는 정합이 지배항."""
    mod = _load()
    model, normals = mod.build_model()
    rng = np.random.default_rng(3)
    T_true = mod.se2(mod.T_TRUE_ANGLE, mod.T_TRUE_XY)
    T_nom = mod.se2(mod.T_NOM_ANGLE, mod.T_NOM_XY)
    probe = mod.sample_probe(rng, T_true, mod.COVERAGE_MAIN, phi0=np.deg2rad(35))
    T_map = mod.se2_inv(mod.register(probe, model, normals)[0])
    entry, _ = mod.plan_entry()

    ct, pd = mod.dyn.ctrl_computed_torque, mod.dyn.ctrl_pd
    a = mod.run_condition(entry, mod.TUMOR_IMG, T_nom, T_true, ct)
    b = mod.run_condition(entry, mod.TUMOR_IMG, T_map, T_true, pd)
    d = mod.run_condition(entry, mod.TUMOR_IMG, T_map, T_true, ct)

    assert a["total"] > 10 * d["total"], "정합이 오차를 줄이지 못함"
    assert b["total"] > 10 * d["total"], "모델기반 제어가 PD보다 낫지 않음"
    # 보정된 계산토크에서는 서보 몫이 정합 몫보다 작다 = 다음 개선 대상은 정합
    assert d["servo"] < d["reg"], "보정 상태인데 서보가 정합보다 큼"


def test_uncalibrated_payload_flips_the_budget():
    """3% 페이로드 오차만으로 서보 몫이 정합 몫을 넘어선다(예산 순위 역전)."""
    mod = _load()
    T = mod.se2(np.deg2rad(8.0), np.array([0.43, 0.05]))
    entry, _ = mod.plan_entry()
    cal = mod.run_condition(entry, mod.TUMOR_IMG, T, T, mod.dyn.ctrl_computed_torque,
                            p_ctrl=mod.ARM)
    unc = mod.run_condition(entry, mod.TUMOR_IMG, T, T, mod.dyn.ctrl_computed_torque,
                            p_ctrl=mod.ARM_CTRL)
    assert unc["servo"] > 100 * max(cal["servo"], 1e-9)
    assert unc["servo"] > 1e-4        # 0.1 mm 이상 = 정합오차 규모를 넘어섬


def test_reliability_gate_catches_bad_registrations():
    """신뢰도 게이트가 unsafe 계획 대부분을 사전 차단하고, 집행분 위험을 크게 낮춘다."""
    mod = _load()
    model, normals = mod.build_model()
    mc = mod.safety_mc(model, normals, n_trials=40, seed=5)

    naive = mc["naive_unsafe"] / mc["n_trials"]
    assert naive > 0.1, "시나리오가 위험을 만들지 못함(테스트 무의미)"
    assert mc["caught"] >= 0.8 * mc["naive_unsafe"], "게이트 검출력 부족"
    if mc["executed"]:
        aware = mc["aware_unsafe"] / mc["executed"]
        assert aware < naive / 2, f"게이트가 집행분 위험을 못 줄임: {aware:.2f} vs {naive:.2f}"


def test_covariance_grows_when_coverage_is_poor():
    """조건화 지표: 표면을 조금만 찍으면 σ_target 이 커진다(정합이 스스로 경고)."""
    mod = _load()
    model, normals = mod.build_model()
    T_true = mod.se2(np.deg2rad(6.0), np.array([0.43, 0.05]))

    sig = {}
    for cov_deg in (90.0, 300.0):
        vals = []
        for s in range(4):
            rng = np.random.default_rng(s)
            probe = mod.sample_probe(rng, T_true, np.deg2rad(cov_deg),
                                     phi0=np.deg2rad(30), n_out=0)
            _, _, Cov, _ = mod.register(probe, model, normals)
            vals.append(mod.target_sigma(Cov, mod.TUMOR_IMG))
        sig[cov_deg] = float(np.median(vals))
    assert sig[90.0] > 2 * sig[300.0], f"커버리지에 따른 σ 변화 없음: {sig}"
