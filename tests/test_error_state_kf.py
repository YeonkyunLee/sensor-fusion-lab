"""오차상태 칼만필터(ESKF) 자세추정 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load():
    spec = importlib.util.spec_from_file_location(
        "eskf37", ROOT / "scripts" / "37_error_state_kf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eskf_beats_gyro_and_estimates_bias():
    m = _load()
    gyro_rmse, eskf_rmse, final_bias_err = m.main(plot=False)

    # (i) ESKF 자세 RMSE가 순수 자이로 적분보다 뚜렷이 작다
    assert eskf_rmse < gyro_rmse * 0.1
    # (ii) 절대 정확도: 몇 도 이내
    assert eskf_rmse < 3.0
    # (iii) 온라인 자이로 바이어스가 참값 근처로 수렴
    assert final_bias_err < 0.02


def test_eskf_stable_across_seeds():
    m = _load()
    for seed in range(4):
        rng = np.random.default_rng(seed)
        t, R_true, bias_true, gyro, accel, mag = m.simulate(rng)
        N = len(t)
        R_eskf, b_eskf = m.run_eskf(gyro, accel, mag)
        err = np.array([m.att_error_deg(R_eskf[k], R_true[k]) for k in range(N)])
        half = N // 2
        eskf_rmse = np.sqrt(np.mean(err[half:] ** 2))
        assert eskf_rmse < 3.0
        assert np.linalg.norm(b_eskf[-1] - bias_true[-1]) < 0.02


def test_heading_tracked_with_magnetometer():
    """지자기를 포함한 완전한 ESKF가 요(헤딩)를 안정적으로 추적한다.
    (관측성: 요는 중력축 회전에 불변이라 가속도만으론 관측 불가 → 지자기가 필요.)"""
    m = _load()
    rng = np.random.default_rng(0)
    t, R_true, bias_true, gyro, accel, mag = m.simulate(rng)
    N = len(t)

    R_full, _ = m.run_eskf(gyro, accel, mag)
    yaw_err = np.degrees(np.abs(np.array([
        np.unwrap([m.rot_to_euler(R_full[k])[2] - m.rot_to_euler(R_true[k])[2]])[0]
        for k in range(N)])))
    # 정상상태 요오차가 작게 유지 (지자기 헤딩 보정 덕분)
    assert yaw_err[N // 2:].mean() < 3.0
