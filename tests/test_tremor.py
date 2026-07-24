"""수술 로봇 생리적 수전증 제거 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tremor_suppressed_motion_preserved():
    mod = _load("22_surgical_tremor.py")
    raw_tremor, filt_tremor, track_err = mod.main()
    # 떨림이 확실히(3배 이상) 억제됨
    assert filt_tremor < raw_tremor * 0.3
    # 의도 움직임 보존: 추종오차가 원래 떨림보다 작게 유지(리칭 진폭 30mm 대비 미미)
    assert track_err < 0.10          # < 100 um
    assert track_err < raw_tremor    # 원신호 떨림/오차보다 작음


def test_flc_beats_kalman_on_tracking():
    mod = _load("22_surgical_tremor.py")
    t, (ix, iy), (rx, ry) = mod.synth(seed=1)
    f0 = mod.estimate_tremor_freq(rx)
    fx, fy = mod.flc(rx, f0), mod.flc(ry, f0)
    kx, ky = mod.kalman_cv(rx), mod.kalman_cv(ry)
    flc_err = mod.tracking_rms(fx, fy, ix, iy)
    kal_err = mod.tracking_rms(kx, ky, ix, iy)
    # 적응 FLC(떨림 대역만 제거)가 등속 칼만(전대역 스무딩)보다 의도동작 추종 우수
    assert flc_err < kal_err
